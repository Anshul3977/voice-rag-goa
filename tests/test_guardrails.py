import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import guardrails  # noqa: E402
from src.retrieval import RetrievedChunk  # noqa: E402


def test_unsafe_input_blocked():
    result = guardrails.check_unsafe_input("how to make a bomb at home")
    assert result.passed is False
    assert result.stage == "unsafe_input"


def test_safe_input_passes():
    result = guardrails.check_unsafe_input("what is the capital of France")
    assert result.passed is True


def test_off_topic_low_score_refused():
    chunks = [RetrievedChunk(chunk_id="a", text="irrelevant", doc_id="1", strategy="fixed", score=0.05)]
    result = guardrails.check_off_topic(chunks)
    assert result.passed is False


def test_off_topic_high_score_passes():
    chunks = [RetrievedChunk(chunk_id="a", text="relevant", doc_id="1", strategy="fixed", score=0.9)]
    result = guardrails.check_off_topic(chunks)
    assert result.passed is True


def test_groundedness_invalid_citation_blocked():
    chunks = [RetrievedChunk(chunk_id="real_id", text="Paris is the capital of France.", doc_id="1", strategy="fixed", score=0.9)]
    result = guardrails.check_groundedness(
        "Paris is the capital.", ["fake_id"], chunks, self_reported_confidence=0.9
    )
    assert result.passed is False
    assert result.stage == "groundedness"


def test_groundedness_good_overlap_passes():
    chunks = [RetrievedChunk(chunk_id="real_id", text="Paris is the capital of France.", doc_id="1", strategy="fixed", score=0.9)]
    result = guardrails.check_groundedness(
        "Paris is the capital of France.", ["real_id"], chunks, self_reported_confidence=0.9
    )
    assert result.passed is True


if __name__ == "__main__":
    test_unsafe_input_blocked()
    test_safe_input_passes()
    test_off_topic_low_score_refused()
    test_off_topic_high_score_passes()
    test_groundedness_invalid_citation_blocked()
    test_groundedness_good_overlap_passes()
    print("All guardrail tests passed.")
