"""
Calls Google Gemini with a strict "answer only from the provided context" prompt
and enforces structured JSON output: {answer, used_chunk_ids, confidence}.
This structured output is what makes the groundedness guardrail possible
(we need the model to tell us which chunks it actually used).
"""
from __future__ import annotations

import json
import os
import re
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from src.retrieval import RetrievedChunk

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are a retrieval-grounded QA assistant. You will be given a user \
question and a numbered list of retrieved context chunks with their chunk_ids. \
Answer ONLY using information present in these chunks. If the chunks do not contain \
enough information to answer, say so explicitly rather than guessing or using outside \
knowledge. Respond with ONLY a JSON object matching this schema exactly:
{"answer": "<string>", "used_chunk_ids": ["<chunk_id>", ...], "confidence": <float 0-1>}
confidence should reflect how directly the retrieved chunks support the answer -- \
use a low value (<0.4) if the context is only tangentially related."""


class GenerationOutput(BaseModel):
    answer: str
    used_chunk_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] chunk_id={c.chunk_id} strategy={c.strategy}\n{c.text}")
    return "\n\n".join(lines)


def _extract_json(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> GenerationOutput:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    context_block = _build_context_block(chunks)
    user_message = f"Question: {query}\n\nContext chunks:\n{context_block}"

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=GenerationOutput,
        temperature=0.1,
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_message,
        config=config,
    )

    raw_text = response.text or ""
    try:
        parsed = _extract_json(raw_text)
        return GenerationOutput(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Model did not return valid structured JSON: {e}\nRaw: {raw_text}")
