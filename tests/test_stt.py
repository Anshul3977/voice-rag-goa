import httpx
import pytest

from src.stt import SpeechNotConfigured, content_type_for, transcribe


def test_webm_content_type():
    assert content_type_for("query.webm") == "audio/webm"
    assert content_type_for("query.wav") == "audio/wav"


def test_transcribe_without_key_does_not_call_groq(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    calls = {"n": 0}

    def explode(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("speech API should not be called")

    monkeypatch.setattr(httpx.Client, "post", explode)
    with pytest.raises(SpeechNotConfigured):
        transcribe(b"fake-bytes", filename="query.webm")
    assert calls["n"] == 0


def test_transcribe_posts_whisper(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("STT_MODEL", raising=False)
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "  what is a corporation  "}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, files=None, data=None):
            seen["url"] = url
            seen["headers"] = headers
            seen["files"] = files
            seen["data"] = data
            return FakeResponse()

    monkeypatch.setattr("src.stt.httpx.Client", FakeClient)
    text = transcribe(b"RIFFfake", filename="query.webm")
    assert text == "what is a corporation"
    assert seen["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["data"]["model"] == "whisper-large-v3-turbo"
    assert seen["files"]["file"][0] == "query.webm"
    assert seen["files"]["file"][2] == "audio/webm"
