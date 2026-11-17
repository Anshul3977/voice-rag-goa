"""
Speech-to-text via Groq's hosted Whisper endpoint.

The free Render instance does not run a local speech model. Audio is posted
to the OpenAI-compatible transcription API. When GROQ_API_KEY is unset the
process still starts; callers get SpeechNotConfigured and the text box works.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"

_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".webm": "audio/webm",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


class SpeechNotConfigured(RuntimeError):
    """GROQ_API_KEY is unset, so microphone audio cannot be transcribed."""


def stt_model() -> str:
    return os.environ.get("STT_MODEL", "").strip() or DEFAULT_STT_MODEL


def stt_status() -> dict:
    configured = bool(os.environ.get("GROQ_API_KEY", "").strip())
    if configured:
        message = f"Speech-to-text: Groq {stt_model()}."
    else:
        message = (
            "GROQ_API_KEY is not set. Type a question instead, "
            "or add the key on the Render service to enable the microphone."
        )
    return {"configured": configured, "model": stt_model(), "message": message}


def content_type_for(filename: str) -> str:
    return _CONTENT_TYPES.get(Path(filename or "").suffix.lower(), "application/octet-stream")


def _retryable_stt_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Return the transcript for raw audio bytes (wav, mp3, webm, and the rest)."""
    status = stt_status()
    if not status["configured"]:
        raise SpeechNotConfigured(status["message"])
    if not audio_bytes:
        raise ValueError("Audio upload was empty.")
    return _transcribe_request(audio_bytes, filename, status["model"])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception(_retryable_stt_error),
    reraise=True,
)
def _transcribe_request(audio_bytes: bytes, filename: str, model: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (filename, audio_bytes, content_type_for(filename))}
    data = {"model": model, "response_format": "json", "temperature": "0"}

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            GROQ_TRANSCRIPTIONS_URL, headers=headers, files=files, data=data
        )
        response.raise_for_status()
        payload = response.json()

    text = payload.get("text") if isinstance(payload, dict) else None
    if text is None:
        raise ValueError("Groq transcription response missing 'text'")
    return str(text).strip()
