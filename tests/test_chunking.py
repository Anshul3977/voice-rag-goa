import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking import build_all_chunks, chunk_fixed, chunk_semantic  # noqa: E402


def test_chunk_fixed_overlap():
    text = " ".join(f"word{i}" for i in range(150))
    chunks = chunk_fixed(text, window_words=60, overlap_words=15)
    assert len(chunks) >= 2
    # verify overlap: last words of chunk[0] should appear at start of chunk[1]
    c0_words = chunks[0].split()
    c1_words = chunks[1].split()
    assert c0_words[-1] in c1_words[: len(c0_words)]


def test_chunk_semantic_respects_sentences():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_semantic(text, max_words=6)
    # no chunk should contain a partial sentence artifact (no dangling comma-less fragment)
    for c in chunks:
        assert c.strip().endswith(".")


def test_build_all_chunks_three_strategies():
    passages = [{"doc_id": "1", "text": "Sentence one. Sentence two is longer here.", "source_query": "q1"}]
    result = build_all_chunks(passages)
    assert set(result.keys()) == {"fixed", "semantic", "metadata_aware"}
    assert all(len(v) >= 1 for v in result.values())
    assert result["metadata_aware"][0]["metadata"]["source_query"] == "q1"


if __name__ == "__main__":
    test_chunk_fixed_overlap()
    test_chunk_semantic_respects_sentences()
    test_build_all_chunks_three_strategies()
    print("All chunking tests passed.")
