import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.orchestrator import Pipeline  # noqa: E402
from src.stt import transcribe  # noqa: E402

STORE_DIR = os.environ.get("DATA_DIR", "./data/store")

app = FastAPI(title="Voice RAG - HH Goa 2026")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_pipeline = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(store_dir=STORE_DIR)
    return _pipeline


class TextQuery(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")))


@app.post("/ask")
async def ask_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    try:
        query_text = transcribe(audio_bytes, filename=audio.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"STT failed: {e}")

    result = get_pipeline().run(query_text)
    return {"transcript": query_text, **result.model_dump()}


@app.post("/ask-text")
async def ask_text(payload: TextQuery):
    """Debug endpoint that skips STT -- useful for testing retrieval/generation
    without recording audio each time."""
    result = get_pipeline().run(payload.query)
    return result.model_dump()


@app.get("/health")
def health():
    return {"status": "ok"}
