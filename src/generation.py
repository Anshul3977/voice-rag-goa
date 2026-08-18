"""
Calls Claude with a strict "answer only from the provided context" prompt
and forces structured JSON output: {answer, used_chunk_ids, confidence}.
This structured output is what makes the groundedness guardrail possible
(we need the model to tell us which chunks it actually used).
"""
from __future__ import annotations

import json
import os
import re
from typing import List

import anthropic
from pydantic import BaseModel, Field, ValidationError

from src.retrieval import RetrievedChunk

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are a retrieval-grounded QA assistant. You will be given a user \
question and a numbered list of retrieved context chunks with their chunk_ids. \
Answer ONLY using information present in these chunks. If the chunks do not contain \
enough information to answer, say so explicitly rather than guessing or using outside \
knowledge. Respond with ONLY a JSON object, no other text, no markdown fences, matching \
this schema exactly:
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
    # Strip markdown fences defensively in case the model adds them anyway.
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> GenerationOutput:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    context_block = _build_context_block(chunks)
    user_message = f"Question: {query}\n\nContext chunks:\n{context_block}"

    resp = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text = "".join(block.text for block in resp.content if block.type == "text")

    try:
        parsed = _extract_json(raw_text)
        return GenerationOutput(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        # Structured-output failure is itself an error the harness should
        # retry/handle -- surface it rather than silently returning raw text.
        raise ValueError(f"Model did not return valid structured JSON: {e}\nRaw: {raw_text}")
