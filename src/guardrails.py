"""
Three guardrail checkpoints in the pipeline:

1. check_unsafe_input(query)         -- before retrieval. Cheap keyword/regex
   screen for self-harm, violence-instruction, and PII-extraction style
   prompts. Fails closed (refuses) rather than guessing intent.

2. check_off_topic(query, retrieved) -- after retrieval, before generation.
   If the best retrieval score is below threshold, the corpus almost
   certainly doesn't cover this query -- refuse instead of letting the LLM
   improvise an ungrounded answer.

3. check_groundedness(answer, retrieved) -- after generation. Verifies the
   model's cited chunk_ids are real, and does a lexical-overlap sanity check
   between the answer text and the cited chunks' text. Combined with the
   model's own self-reported confidence, decides whether to surface the
   answer or fall back to "not enough information".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.retrieval import RetrievedChunk

UNSAFE_PATTERNS = [
    r"\bhow (to|do i) (make|build|synthesi[sz]e) (a )?(bomb|explosive|weapon)\b",
    r"\bhow (to|do i) (kill|hurt|harm) (myself|someone)\b",
    r"\bsuicide method\b",
    r"\bsocial security number\b",
    r"\bcredit card number\b",
    r"\bself[- ]harm\b",
]
_UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]

OFF_TOPIC_SCORE_THRESHOLD = 0.30  # cosine sim on normalized MiniLM embeddings
GROUNDEDNESS_OVERLAP_THRESHOLD = 0.15  # fraction of answer words present in cited chunks


@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    stage: Optional[str] = None


def check_unsafe_input(query: str) -> GuardrailResult:
    for pattern in _UNSAFE_RE:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                reason="Query matched an unsafe-input pattern; refusing before retrieval.",
                stage="unsafe_input",
            )
    return GuardrailResult(passed=True)


def check_off_topic(retrieved: List[RetrievedChunk]) -> GuardrailResult:
    if not retrieved or retrieved[0].score < OFF_TOPIC_SCORE_THRESHOLD:
        return GuardrailResult(
            passed=False,
            reason=(
                "Best retrieval similarity is below threshold "
                f"({retrieved[0].score:.2f} < {OFF_TOPIC_SCORE_THRESHOLD})"
                if retrieved
                else "No chunks retrieved."
            ),
            stage="off_topic",
        )
    return GuardrailResult(passed=True)


def _word_overlap_ratio(answer: str, source_text: str) -> float:
    answer_words = set(re.findall(r"\w+", answer.lower()))
    source_words = set(re.findall(r"\w+", source_text.lower()))
    if not answer_words:
        return 0.0
    return len(answer_words & source_words) / len(answer_words)


def check_groundedness(
    answer_text: str,
    cited_chunk_ids: List[str],
    retrieved: List[RetrievedChunk],
    self_reported_confidence: float,
) -> GuardrailResult:
    retrieved_ids = {c.chunk_id for c in retrieved}
    invalid_citations = [cid for cid in cited_chunk_ids if cid not in retrieved_ids]
    if invalid_citations:
        return GuardrailResult(
            passed=False,
            reason=f"Model cited chunk_ids not present in retrieved set: {invalid_citations}",
            stage="groundedness",
        )

    cited_text = " ".join(c.text for c in retrieved if c.chunk_id in cited_chunk_ids)
    if not cited_text:
        cited_text = " ".join(c.text for c in retrieved)

    overlap = _word_overlap_ratio(answer_text, cited_text)
    if overlap < GROUNDEDNESS_OVERLAP_THRESHOLD or self_reported_confidence < 0.4:
        return GuardrailResult(
            passed=False,
            reason=(
                f"Low groundedness: word_overlap={overlap:.2f} "
                f"(min {GROUNDEDNESS_OVERLAP_THRESHOLD}), "
                f"model_confidence={self_reported_confidence:.2f} (min 0.4)"
            ),
            stage="groundedness",
        )
    return GuardrailResult(passed=True)
