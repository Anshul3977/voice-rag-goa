"""
The harness. Every stage is:
  - given typed input
  - wrapped in try/except with retry where the failure could be transient
  - timed individually
  - allowed to short-circuit the pipeline with a typed, user-safe response
    (guardrail refusals, STT failures, generation failures) instead of a
    raw stack trace reaching the caller.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src import guardrails
from src.generation import GenerationOutput, generate_answer
from src.retrieval import Retriever


class PipelineStatus(str, Enum):
    OK = "ok"
    REFUSED_UNSAFE = "refused_unsafe"
    REFUSED_OFF_TOPIC = "refused_off_topic"
    REFUSED_UNGROUNDED = "refused_ungrounded"
    ERROR = "error"


class ChunkCitation(BaseModel):
    chunk_id: str
    strategy: str
    score: float
    text: str


class PipelineResult(BaseModel):
    status: PipelineStatus
    query: str
    answer: Optional[str] = None
    confidence: Optional[float] = None
    citations: List[ChunkCitation] = []
    refusal_reason: Optional[str] = None
    timings_ms: dict = {}
    error: Optional[str] = None


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=3))
def _generate_with_retry(query: str, chunks):
    return generate_answer(query, chunks)


class Pipeline:
    """End-to-end orchestrator: query text -> PipelineResult.
    (STT happens upstream in app/server.py, which then calls run().)
    """

    def __init__(self, store_dir: str, top_n: int = 5):
        self.retriever = Retriever(store_dir)
        self.top_n = top_n

    def run(self, query: str) -> PipelineResult:
        timings = {}
        t_start = time.perf_counter()

        # --- Guardrail 1: unsafe input, before any retrieval/LLM spend ---
        t0 = time.perf_counter()
        unsafe_check = guardrails.check_unsafe_input(query)
        timings["guardrail_unsafe_ms"] = (time.perf_counter() - t0) * 1000
        if not unsafe_check.passed:
            return PipelineResult(
                status=PipelineStatus.REFUSED_UNSAFE,
                query=query,
                refusal_reason=unsafe_check.reason,
                timings_ms=timings,
            )

        # --- Retrieval ---
        try:
            retrieved, retrieval_timings = self.retriever.retrieve(query, top_n=self.top_n)
            timings.update(retrieval_timings)
        except Exception as e:
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            return PipelineResult(
                status=PipelineStatus.ERROR, query=query, error=str(e), timings_ms=timings
            )

        # --- Guardrail 2: off-topic ---
        t1 = time.perf_counter()
        off_topic_check = guardrails.check_off_topic(retrieved)
        timings["guardrail_off_topic_ms"] = (time.perf_counter() - t1) * 1000
        if not off_topic_check.passed:
            timings["total_retrieval_side_ms"] = (time.perf_counter() - t_start) * 1000
            return PipelineResult(
                status=PipelineStatus.REFUSED_OFF_TOPIC,
                query=query,
                refusal_reason=off_topic_check.reason,
                timings_ms=timings,
            )

        timings["total_retrieval_side_ms"] = (time.perf_counter() - t_start) * 1000

        # --- Generation (LLM network call -- reported separately from
        # the retrieval-side latency budget, see README section 2) ---
        t2 = time.perf_counter()
        try:
            gen: GenerationOutput = _generate_with_retry(query, retrieved)
        except Exception as e:
            timings["generation_ms"] = (time.perf_counter() - t2) * 1000
            return PipelineResult(
                status=PipelineStatus.ERROR, query=query, error=str(e), timings_ms=timings
            )
        timings["generation_ms"] = (time.perf_counter() - t2) * 1000

        # --- Guardrail 3: groundedness / hallucination check ---
        t3 = time.perf_counter()
        grounded_check = guardrails.check_groundedness(
            gen.answer, gen.used_chunk_ids, retrieved, gen.confidence
        )
        timings["guardrail_groundedness_ms"] = (time.perf_counter() - t3) * 1000
        if not grounded_check.passed:
            return PipelineResult(
                status=PipelineStatus.REFUSED_UNGROUNDED,
                query=query,
                refusal_reason=grounded_check.reason,
                timings_ms=timings,
            )

        citations = [
            ChunkCitation(chunk_id=c.chunk_id, strategy=c.strategy, score=c.score, text=c.text)
            for c in retrieved
            if c.chunk_id in gen.used_chunk_ids
        ] or [
            ChunkCitation(chunk_id=c.chunk_id, strategy=c.strategy, score=c.score, text=c.text)
            for c in retrieved
        ]

        return PipelineResult(
            status=PipelineStatus.OK,
            query=query,
            answer=gen.answer,
            confidence=gen.confidence,
            citations=citations,
            timings_ms=timings,
        )
