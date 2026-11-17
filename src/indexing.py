"""
Loads the committed chunk metadata for each strategy.

The free Render instance has 512 MB of RAM. A multilingual MiniLM ONNX
session used about 680 MB in a local check, so this service does not load a
transformer or FAISS. Retrieval scores the same chunks with BM25.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0900-\u0963\u0970-\u097F]+")

# Function words that would otherwise dominate BM25 in this Hindi/English mix.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
    "का", "के", "की", "को", "में", "से", "पर", "और", "या", "एक", "यह", "ये",
    "क्या", "जो", "तो", "भी", "ने", "है", "हैं", "था", "थी", "थे", "हो",
    "नहीं", "लिए", "कर", "कि", "इस", "उस", "वे", "वह",
}


def tokenize(text: str) -> List[str]:
    """Keep Devanagari letters and combining marks inside a word.

    Python's ``\\w`` splits Hindi words on vowel signs, which throws away the
    term a BM25 lookup needs. Function words are dropped so a question like
    "what is X" is scored on X.
    """
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def build_indices(chunk_sets: Dict[str, List[Dict]], store_dir: str) -> None:
    """Write chunk metadata. Scoring is BM25 over these files, not a vector index."""
    os.makedirs(store_dir, exist_ok=True)
    for strategy, chunks in chunk_sets.items():
        if not chunks:
            continue
        path = os.path.join(store_dir, f"{strategy}_meta.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(chunks, handle, ensure_ascii=False)
        print(f"  Wrote {len(chunks)} chunks for strategy '{strategy}'", flush=True)


def index_files_present(store_dir: str) -> bool:
    needed = ("metadata_aware_meta.json", "fixed_meta.json")
    return all(os.path.exists(os.path.join(store_dir, name)) for name in needed)


def load_chunks(strategy: str, store_dir: str) -> List[Dict]:
    meta_path = os.path.join(store_dir, f"{strategy}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing chunk metadata for strategy '{strategy}' in {store_dir}. "
            "Run data/prepare_dataset.py first."
        )
    with open(meta_path, "r", encoding="utf-8") as handle:
        raw_chunks = json.load(handle)
    chunks = []
    for chunk in raw_chunks:
        meta = chunk.get("metadata", {})
        pruned = {
            key: meta[key]
            for key in ("is_selected", "source_query", "query_id", "position")
            if key in meta
        }
        chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "metadata": pruned,
            }
        )
    return chunks
