"""
Three deliberately different chunking strategies, so retrieval isn't betting
on one splitting philosophy:

1. fixed        — fixed token/word window with overlap. Cheap, high recall,
                   good for short MS MARCO-style passages where a passage IS
                   basically already a good chunk.
2. semantic      — sentence-boundary aware, greedily packs sentences up to a
                   token budget so we never cut a sentence mid-thought.
                   Better precision for factoid QA.
3. metadata_aware— same as semantic, but keeps + surfaces provenance
                   (doc_id, source_query, position) as first-class metadata
                   used later for filtering/boosting in retrieval and for
                   citation in the final answer.

Each chunk is a dict: {chunk_id, text, doc_id, strategy, metadata}
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

WORD_RE = re.compile(r"\S+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # includes Devanagari danda '।' for Hindi text


def _chunk_id(doc_id: str, strategy: str, idx: int) -> str:
    raw = f"{doc_id}:{strategy}:{idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def chunk_fixed(text: str, window_words: int = 60, overlap_words: int = 15) -> List[str]:
    """Fixed-size sliding window over words, with overlap to avoid losing
    context at boundaries."""
    words = WORD_RE.findall(text)
    if not words:
        return []
    step = max(window_words - overlap_words, 1)
    out = []
    for start in range(0, len(words), step):
        window = words[start : start + window_words]
        if not window:
            break
        out.append(" ".join(window))
        if start + window_words >= len(words):
            break
    return out


def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else ([text.strip()] if text.strip() else [])


def chunk_semantic(text: str, max_words: int = 80) -> List[str]:
    """Greedily pack whole sentences up to a word budget — never splits a
    sentence, which fixed-window chunking can do."""
    sentences = split_sentences(text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(WORD_RE.findall(sent))
        if current and current_len + sent_len > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_all_chunks(passages: List[Dict]) -> Dict[str, List[Dict]]:
    """
    passages: [{doc_id, text, source_query}, ...]
    returns: {"fixed": [...chunks...], "semantic": [...], "metadata_aware": [...]}
    """
    result = {"fixed": [], "semantic": [], "metadata_aware": []}

    for p in passages:
        doc_id, text, source_query = p["doc_id"], p["text"], p.get("source_query", "")
        is_selected = p.get("is_selected", False)
        query_id = p.get("query_id")

        for idx, ch in enumerate(chunk_fixed(text)):
            result["fixed"].append(
                {
                    "chunk_id": _chunk_id(doc_id, "fixed", idx),
                    "text": ch,
                    "doc_id": doc_id,
                    "strategy": "fixed",
                    "metadata": {"position": idx},
                }
            )

        for idx, ch in enumerate(chunk_semantic(text)):
            result["semantic"].append(
                {
                    "chunk_id": _chunk_id(doc_id, "semantic", idx),
                    "text": ch,
                    "doc_id": doc_id,
                    "strategy": "semantic",
                    "metadata": {"position": idx},
                }
            )

        # metadata_aware reuses the semantic split (best base granularity)
        # but attaches provenance fields that retrieval/generation can use
        # for filtering, boosting, and citation.
        for idx, ch in enumerate(chunk_semantic(text, max_words=80)):
            result["metadata_aware"].append(
                {
                    "chunk_id": _chunk_id(doc_id, "metadata_aware", idx),
                    "text": ch,
                    "doc_id": doc_id,
                    "strategy": "metadata_aware",
                    "metadata": {
                        "position": idx,
                        "source_query": source_query,
                        "query_id": query_id,
                        "is_selected": is_selected,
                        "char_len": len(ch),
                        "word_len": len(WORD_RE.findall(ch)),
                    },
                }
            )

    return result
