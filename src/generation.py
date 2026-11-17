"""
Grounded answer generation through an OpenAI-compatible chat API.

Groq is the default provider (https://api.groq.com/openai/v1). Set
LLM_PROVIDER=openai to use OpenAI or any other compatible base URL
(OPENAI_BASE_URL / LLM_BASE_URL). The model id comes from GENERATION_MODEL,
otherwise a provider default is used.

The model must return JSON: {answer, used_chunk_ids, confidence}. That
structure is what the groundedness guardrail checks.
"""
from __future__ import annotations

import json
import os
import re
from typing import List

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.retrieval import RetrievedChunk

DEFAULT_PROVIDER = "groq"

# OpenAI-compatible providers. Keys are read at call time so tests and
# Render env vars can change them without restarting the import.
PROVIDER_SETTINGS = {
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
}

SYSTEM_PROMPT = """You are a retrieval-grounded QA assistant. You will be given a user \
question and a list of retrieved context chunks with their Chunk IDs. \
Answer ONLY using information present in these chunks. If the chunks do not contain \
enough information to answer, say so explicitly rather than guessing or using outside \
knowledge. Write the answer in the same language as the context chunks and reuse their \
wording where you can, so the answer stays lexically grounded in the cited text. \
Respond with ONLY a JSON object matching this schema exactly:
{"answer": "<string>", "used_chunk_ids": ["<exact Chunk ID string from context>", ...], "confidence": <float between 0.0 and 1.0>}
confidence should reflect how directly the retrieved chunks support the answer -- \
use a low value (<0.4) if the context is only tangentially related."""


class LLMNotConfigured(RuntimeError):
    """The selected provider has no API key."""


class TransientLLMError(RuntimeError):
    """A retryable network or rate-limit failure."""


class GenerationOutput(BaseModel):
    answer: str
    used_chunk_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @field_validator("used_chunk_ids")
    @classmethod
    def _clean_ids(cls, value: List[str]) -> List[str]:
        return [str(item).strip() for item in value if str(item).strip()]


def llm_status() -> dict:
    """Describe the configured provider without calling it."""
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    if provider not in PROVIDER_SETTINGS:
        return {
            "provider": provider,
            "configured": False,
            "model": None,
            "base_url": None,
            "key_env": None,
            "message": (
                f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' (default) or 'openai'."
            ),
        }

    spec = PROVIDER_SETTINGS[provider]
    key = os.environ.get(spec["key_env"], "").strip()
    base_url = _base_url(provider, spec["base_url"])
    model = os.environ.get("GENERATION_MODEL", "").strip() or spec["default_model"]
    if key:
        message = f"Language model: {provider} ({model})."
    else:
        message = (
            f"{spec['key_env']} is not set. The app is in retrieval-only mode: "
            "questions still return cited passages, and no language model is called. "
            f"Add {spec['key_env']} on the Render service to generate a grounded answer."
        )
    return {
        "provider": provider,
        "configured": bool(key),
        "model": model,
        "base_url": base_url,
        "key_env": spec["key_env"],
        "message": message,
    }


def _base_url(provider: str, default: str) -> str:
    override = os.environ.get("LLM_BASE_URL", "").strip()
    if provider == "openai":
        override = os.environ.get("OPENAI_BASE_URL", "").strip() or override
    return (override or default).rstrip("/")


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for chunk in chunks:
        lines.append(
            f"Chunk ID: {chunk.chunk_id} (Strategy: {chunk.strategy})\nText: {chunk.text}"
        )
    return "\n\n".join(lines)


def _extract_json(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected a JSON object", cleaned, 0)
    return parsed


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> GenerationOutput:
    status = llm_status()
    if not status["configured"]:
        raise LLMNotConfigured(status["message"])

    api_key = os.environ.get(status["key_env"], "").strip()
    context_block = _build_context_block(chunks)
    payload = {
        "model": status["model"],
        "temperature": 0.1,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext chunks:\n{context_block}",
            },
        ],
    }
    url = f"{status['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 429 or code >= 500:
            raise TransientLLMError(f"LLM request failed with HTTP {code}") from exc
        raise RuntimeError(f"LLM request failed with HTTP {code}") from exc
    except httpx.TransportError as exc:
        raise TransientLLMError(f"LLM request failed: {exc.__class__.__name__}") from exc

    try:
        body = response.json()
        raw_text = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("LLM response did not include a message") from exc

    try:
        parsed = _extract_json(raw_text)
        return GenerationOutput(**parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Model did not return valid structured JSON: {exc}") from exc
