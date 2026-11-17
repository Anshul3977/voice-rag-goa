from fastapi.testclient import TestClient

from app.server import app
from src.orchestrator import Pipeline, PipelineStatus
from src.retrieval import RetrievedChunk
from tests.test_orchestrator import FakeRetriever


def _client():
    return TestClient(app)


def test_health_without_keys(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is False
    assert body["stt_configured"] is False
    assert body["index_ready"] is True
    assert "GROQ_API_KEY" in body["message"]


def test_home_page_renders():
    response = _client().get("/")
    assert response.status_code == 200
    assert "Voice RAG" in response.text
    assert "Read aloud" in response.text


def test_ask_without_key_explains_and_skips_speech(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def explode(*args, **kwargs):
        raise AssertionError("transcribe should not run")

    monkeypatch.setattr("app.server.transcribe", explode)
    response = _client().post(
        "/ask", files={"audio": ("query.webm", b"not-audio", "audio/webm")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stt_unconfigured"
    assert "GROQ_API_KEY" in body["error"]


def test_ask_text_retrieval_only(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="निगम कानून द्वारा बनाया गया व्यक्तियों का समूह है।",
        doc_id="d1",
        strategy="metadata_aware",
        score=12.0,
    )
    pipeline = Pipeline(store_dir=".", retriever=FakeRetriever([chunk]))
    monkeypatch.setattr("app.server.get_pipeline", lambda: pipeline)
    response = _client().post("/ask-text", json={"query": "कॉर्पोरेशन क्या है?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == PipelineStatus.RETRIEVAL_ONLY.value
    assert body["citations"][0]["chunk_id"] == "c1"
