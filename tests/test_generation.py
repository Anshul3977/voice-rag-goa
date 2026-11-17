import json

import httpx
import pytest

from src.generation import (
    LLMNotConfigured,
    TransientLLMError,
    _extract_json,
    generate_answer,
    llm_status,
)
from src.retrieval import RetrievedChunk


def _chunk():
    return RetrievedChunk(
        chunk_id="c1",
        text="Paris is the capital of France.",
        doc_id="d1",
        strategy="fixed",
        score=0.9,
    )


def test_extract_json_strips_fences():
    raw = '```json\n{"answer": "Paris", "used_chunk_ids": ["c1"], "confidence": 0.8}\n```'
    parsed = _extract_json(raw)
    assert parsed["answer"] == "Paris"


def test_default_provider_is_groq_and_unconfigured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    status = llm_status()
    assert status["provider"] == "groq"
    assert status["configured"] is False
    assert status["model"] == "llama-3.3-70b-versatile"
    assert "GROQ_API_KEY" in status["message"]


def test_generate_posts_groq_chat_completion(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Paris is the capital of France.",
                                    "used_chunk_ids": ["c1"],
                                    "confidence": 0.91,
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            seen["url"] = url
            seen["headers"] = headers
            seen["json"] = json
            return FakeResponse()

    monkeypatch.setattr("src.generation.httpx.Client", FakeClient)
    output = generate_answer("capital of France?", [_chunk()])
    assert output.answer.startswith("Paris")
    assert output.used_chunk_ids == ["c1"]
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["json"]["model"] == "llama-3.3-70b-versatile"


def test_missing_key_does_not_call_the_network(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    def explode(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr(httpx.Client, "post", explode)
    with pytest.raises(LLMNotConfigured):
        generate_answer("capital?", [_chunk()])


def test_rate_limit_is_transient(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeResponse:
        status_code = 429

        def raise_for_status(self):
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            raise httpx.HTTPStatusError("rate limit", request=request, response=self)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("src.generation.httpx.Client", FakeClient)
    with pytest.raises(TransientLLMError):
        generate_answer("capital?", [_chunk()])
