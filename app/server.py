import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.generation import llm_status
from src.indexing import index_files_present
from src.orchestrator import Pipeline
from src.stt import SpeechNotConfigured, stt_status, transcribe

load_dotenv()

STORE_DIR = os.environ.get("DATA_DIR", "./data/store")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _warm_pipeline() -> None:
    try:
        get_pipeline()
    except Exception:
        return


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The socket opens as soon as this returns. Model load stays off the health check.
    if os.environ.get("WARM_START", "0") == "1" and index_files_present(STORE_DIR):
        threading.Thread(target=_warm_pipeline, name="warm-embedder", daemon=True).start()
    yield


app = FastAPI(title="Voice RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_error = None


def get_pipeline() -> Pipeline:
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            try:
                _pipeline = Pipeline(store_dir=STORE_DIR)
                _pipeline_error = None
            except Exception as exc:
                _pipeline_error = str(exc)
                raise
    return _pipeline


class TextQuery(BaseModel):
    query: str = Field(min_length=1)


def _status_payload() -> dict:
    provider = llm_status()
    speech = stt_status()
    indexes_ready = index_files_present(STORE_DIR)
    if not indexes_ready:
        message = (
            "The chunk files are missing from DATA_DIR. "
            "The committed store should ship with the service."
        )
    elif _pipeline_error:
        message = f"The retriever failed to load: {_pipeline_error}"
    elif not provider["configured"]:
        message = provider["message"]
    else:
        message = f"{provider['message']} {speech['message']}"
    return {
        "status": "ok" if indexes_ready and not _pipeline_error else "error",
        "llm_provider": provider["provider"],
        "llm_model": provider["model"],
        "llm_configured": provider["configured"],
        "stt_model": speech["model"],
        "stt_configured": speech["configured"],
        "index_ready": indexes_ready,
        "index_loaded": _pipeline is not None,
        "message": message,
    }


@app.get("/health")
def health():
    payload = _status_payload()
    # Missing API keys are a supported degraded mode, not a failed deploy.
    if not payload["index_ready"] or _pipeline_error:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/ask")
async def ask_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.webm"
    if not stt_status()["configured"]:
        return {
            "transcript": "",
            "status": "stt_unconfigured",
            "error": stt_status()["message"],
        }
    try:
        query_text = transcribe(audio_bytes, filename=filename)
    except SpeechNotConfigured as exc:
        return {"transcript": "", "status": "stt_unconfigured", "error": str(exc)}
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "transcript": "",
                "status": "error",
                "error": f"Speech-to-text failed ({exc.__class__.__name__}).",
            },
        )
    if not query_text:
        return {
            "transcript": "",
            "status": "error",
            "error": "The recording did not contain any words.",
        }
    try:
        result = get_pipeline().run(query_text)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"transcript": query_text, "status": "error", "error": str(exc)},
        )
    return {"transcript": query_text, **result.model_dump(mode="json")}


@app.post("/ask-text")
async def ask_text(payload: TextQuery):
    """Skip speech-to-text. Used by the text box and by tests."""
    try:
        result = get_pipeline().run(payload.query)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "query": payload.query, "error": str(exc)},
        )
    return result.model_dump(mode="json")
