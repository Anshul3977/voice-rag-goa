"""
Retrieves from the fixed-window and metadata-aware chunk sets and merges them.

Semantic chunks live in the metadata-aware set, so they are not searched
twice. Scores are BM25, weighted (metadata-aware favored, fixed included for
recall) and deduplicated by doc_id before the final top_n.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

STRATEGY_WEIGHTS = {
    "metadata_aware": 1.0,
    "fixed": 0.85,
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_id: str
    strategy: str
    score: float
    metadata: dict = field(default_factory=dict)


class _BM25:
    def __init__(self, chunks: List[dict], k1: float = 1.5, b: float = 0.75):
        from src.indexing import tokenize

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[tuple]] = defaultdict(list)
        for doc_id, chunk in enumerate(chunks):
            counts: Dict[str, int] = defaultdict(int)
            tokens = tokenize(chunk["text"])
            for token in tokens:
                counts[token] += 1
            self.doc_len.append(len(tokens) or 1)
            for token, freq in counts.items():
                self.postings[token].append((doc_id, freq))
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self.n_docs = len(chunks)

    def scores(self, query_tokens: List[str]) -> Dict[int, float]:
        totals: Dict[int, float] = defaultdict(float)
        for token in set(query_tokens):
            hits = self.postings.get(token)
            if not hits:
                continue
            df = len(hits)
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for doc_id, freq in hits:
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                totals[doc_id] += idf * (freq * (self.k1 + 1)) / denom
        return totals


class Retriever:
    def __init__(self, store_dir: str):
        from src.indexing import load_chunks

        self.store_dir = store_dir
        self._loaded = {}
        for strategy in STRATEGY_WEIGHTS:
            try:
                self._loaded[strategy] = _BM25(load_chunks(strategy, store_dir))
            except FileNotFoundError:
                pass
        if not self._loaded:
            raise RuntimeError(
                f"No chunk metadata found in {store_dir}. Run data/prepare_dataset.py first."
            )

    def retrieve(self, query: str, top_k_per_strategy: int = 5, top_n: int = 5):
        from src.indexing import tokenize

        t0 = time.perf_counter()
        query_tokens = tokenize(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        candidates: List[RetrievedChunk] = []
        for strategy, index in self._loaded.items():
            ranked = index.scores(query_tokens)
            if not ranked:
                continue
            top = sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:top_k_per_strategy]
            for doc_id, score in top:
                chunk = index.chunks[doc_id]
                weighted = float(score) * STRATEGY_WEIGHTS.get(strategy, 1.0)
                candidates.append(
                    RetrievedChunk(
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        doc_id=chunk["doc_id"],
                        strategy=strategy,
                        score=weighted,
                        metadata=chunk.get("metadata", {}),
                    )
                )

        # Deduplicate by doc_id, keeping the highest-scoring chunk per doc
        # so we don't return 3 near-identical chunks of the same passage.
        best_per_doc: Dict[str, RetrievedChunk] = {}
        for c in candidates:
            existing = best_per_doc.get(c.doc_id)
            if existing is None or c.score > existing.score:
                best_per_doc[c.doc_id] = c

        merged = sorted(best_per_doc.values(), key=lambda c: c.score, reverse=True)[:top_n]
        retrieve_ms = (time.perf_counter() - t1) * 1000

        return merged, {"embed_query_ms": embed_ms, "retrieve_ms": retrieve_ms}
