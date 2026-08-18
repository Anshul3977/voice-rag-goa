import os
from dotenv import load_dotenv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    audio_bytes: raw audio file bytes (wav/mp3/webm etc.)
    returns: transcribed text
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set.")

    headers = {"xi-api-key": api_key}
    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"model_id": "scribe_v1"}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data)
        resp.raise_for_status()
        res_data = resp.json()

    text = res_data.get("text")
    if text is None:
        raise ValueError(f"ElevenLabs STT response missing 'text': {res_data}")
    return text.strip()
