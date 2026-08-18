"""
Retrieves from all three strategy indices and merges results, rather than
committing to a single chunking philosophy. Each strategy contributes
top-k candidates; scores are weighted (semantic/metadata_aware slightly
favored for precision, fixed included for recall) then deduplicated by
underlying doc_id before returning the final top_n.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

from src.indexing import embed_texts, load_index

STRATEGY_WEIGHTS = {
    "semantic": 1.0,
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


class Retriever:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self._loaded = {}
        for strategy in STRATEGY_WEIGHTS:
            try:
                self._loaded[strategy] = load_index(strategy, store_dir)
            except FileNotFoundError:
                pass
        if not self._loaded:
            raise RuntimeError(
                f"No indices found in {store_dir}. Run data/prepare_dataset.py first."
            )

    def retrieve(self, query: str, top_k_per_strategy: int = 5, top_n: int = 5):
        t0 = time.perf_counter()
        q_vec = embed_texts([query])
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        candidates: List[RetrievedChunk] = []
        for strategy, (index, chunks) in self._loaded.items():
            k = min(top_k_per_strategy, index.ntotal)
            if k == 0:
                continue
            scores, idxs = index.search(q_vec, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                c = chunks[idx]
                weighted = float(score) * STRATEGY_WEIGHTS.get(strategy, 1.0)
                candidates.append(
                    RetrievedChunk(
                        chunk_id=c["chunk_id"],
                        text=c["text"],
                        doc_id=c["doc_id"],
                        strategy=strategy,
                        score=weighted,
                        metadata=c.get("metadata", {}),
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
