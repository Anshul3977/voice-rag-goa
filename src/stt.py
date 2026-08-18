"""
Speech-to-text via ElevenLabs. Wrapped with retry/backoff since STT is a
network call and the harness treats all external calls as fallible.
"""
from __future__ import annotations

import io
import os

from elevenlabs.client import ElevenLabs
from tenacity import retry, stop_after_attempt, wait_exponential

_client = None


def get_client() -> ElevenLabs:
    global _client
    if _client is None:
        _client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    audio_bytes: raw audio file bytes (wav/mp3/webm etc.)
    returns: transcribed text
    """
    client = get_client()
    result = client.speech_to_text.convert(
        file=io.BytesIO(audio_bytes),
        model_id="scribe_v1",
    )
    text = getattr(result, "text", None)
    if not text:
        raise ValueError("ElevenLabs STT returned no text.")
    return text.strip()
