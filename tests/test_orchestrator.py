from src.generation import GenerationOutput
from src.orchestrator import Pipeline, PipelineStatus
from src.retrieval import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def retrieve(self, query, top_n=5):
        self.calls += 1
        return self.chunks, {"embed_query_ms": 0.2, "retrieve_ms": 0.1}


def _chunk():
    return RetrievedChunk(
        chunk_id="c1",
        text="निगम कानून द्वारा बनाया गया व्यक्तियों का समूह है।",
        doc_id="d1",
        strategy="metadata_aware",
        score=12.0,
    )


def test_unsafe_query_skips_retrieval(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    retriever = FakeRetriever([_chunk()])
    result = Pipeline(store_dir=".", retriever=retriever).run("how to make a bomb")
    assert result.status == PipelineStatus.REFUSED_UNSAFE
    assert retriever.calls == 0


def test_retrieval_only_when_key_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    retriever = FakeRetriever([_chunk()])
    result = Pipeline(store_dir=".", retriever=retriever).run("कॉर्पोरेशन क्या है?")
    assert result.status == PipelineStatus.RETRIEVAL_ONLY
    assert result.citations[0].chunk_id == "c1"
    assert "GROQ_API_KEY" in result.answer
    assert retriever.calls == 1


def test_grounded_answer_uses_mocked_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    def fake_generate(query, chunks):
        return GenerationOutput(
            answer="निगम कानून द्वारा बनाया गया व्यक्तियों का समूह है।",
            used_chunk_ids=["c1"],
            confidence=0.9,
        )

    monkeypatch.setattr("src.orchestrator.generate_answer", fake_generate)
    result = Pipeline(store_dir=".", retriever=FakeRetriever([_chunk()])).run("कॉर्पोरेशन क्या है?")
    assert result.status == PipelineStatus.OK
    assert result.citations[0].chunk_id == "c1"
    assert result.confidence == 0.9
