"""
The harness. Every stage is:
  - given typed input
  - wrapped in try/except with retry where the failure could be transient
  - timed individually
  - allowed to short-circuit the pipeline with a typed, user-safe response
    (guardrail refusals, missing LLM key, generation failures) instead of a
    raw stack trace reaching the caller.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import guardrails
from src.generation import GenerationOutput, TransientLLMError, generate_answer, llm_status
from src.retrieval import RetrievedChunk, Retriever


class PipelineStatus(str, Enum):
    OK = "ok"
    RETRIEVAL_ONLY = "retrieval_only"
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


def _citations(chunks: List[RetrievedChunk], chunk_ids: Optional[List[str]] = None) -> List[ChunkCitation]:
    selected = chunks
    if chunk_ids:
        matched = [chunk for chunk in chunks if chunk.chunk_id in chunk_ids]
        if matched:
            selected = matched
    return [
        ChunkCitation(
            chunk_id=chunk.chunk_id,
            strategy=chunk.strategy,
            score=chunk.score,
            text=chunk.text,
        )
        for chunk in selected
    ]


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    retry=retry_if_exception_type(TransientLLMError),
    reraise=True,
)
def _generate_with_retry(query: str, chunks):
    return generate_answer(query, chunks)


class Pipeline:
    """End-to-end orchestrator: query text -> PipelineResult.

    Speech-to-text happens upstream in app/server.py, which then calls run().
    """

    def __init__(self, store_dir: str, top_n: int = 5, retriever=None):
        self.retriever = retriever if retriever is not None else Retriever(store_dir)
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
        except Exception as exc:
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            return PipelineResult(
                status=PipelineStatus.ERROR, query=query, error=str(exc), timings_ms=timings
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

        # No key: still return the passages the retriever found.
        provider = llm_status()
        if not provider["configured"]:
            return PipelineResult(
                status=PipelineStatus.RETRIEVAL_ONLY,
                query=query,
                answer=provider["message"],
                citations=_citations(retrieved),
                timings_ms=timings,
            )

        # --- Generation (LLM network call -- reported separately from
        # the retrieval-side latency budget) ---
        t2 = time.perf_counter()
        try:
            generated: GenerationOutput = _generate_with_retry(query, retrieved)
        except Exception as exc:
            timings["generation_ms"] = (time.perf_counter() - t2) * 1000
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            return PipelineResult(
                status=PipelineStatus.ERROR, query=query, error=str(exc), timings_ms=timings
            )
        timings["generation_ms"] = (time.perf_counter() - t2) * 1000

        # --- Guardrail 3: groundedness / hallucination check ---
        t3 = time.perf_counter()
        grounded_check = guardrails.check_groundedness(
            generated.answer, generated.used_chunk_ids, retrieved, generated.confidence
        )
        timings["guardrail_groundedness_ms"] = (time.perf_counter() - t3) * 1000
        if not grounded_check.passed:
            return PipelineResult(
                status=PipelineStatus.REFUSED_UNGROUNDED,
                query=query,
                refusal_reason=grounded_check.reason,
                timings_ms=timings,
                citations=_citations(retrieved, generated.used_chunk_ids),
            )

        return PipelineResult(
            status=PipelineStatus.OK,
            query=query,
            answer=generated.answer,
            confidence=generated.confidence,
            citations=_citations(retrieved, generated.used_chunk_ids),
            timings_ms=timings,
        )
