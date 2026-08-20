"""
Builds one FAISS IndexFlatIP (cosine sim via L2-normalized embeddings) per
chunking strategy, and persists chunk metadata alongside so retrieval can
map vector hits back to text + provenance.
"""
from __future__ import annotations

import gc
import json
import os
from typing import Dict, List

import faiss
import numpy as np

import torch
from sentence_transformers import SentenceTransformer

# Limit PyTorch thread pool memory overhead
torch.set_num_threads(4)

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME).half()
    return _model


def embed_texts(texts: List[str], chunk_size: int = 512) -> np.ndarray:
    model = get_embedder()
    all_vecs = []
    
    # Process in sub-batches to prevent PyTorch memory spikes
    for i in range(0, len(texts), chunk_size):
        sub_texts = texts[i:i + chunk_size]
        sub_vecs = model.encode(sub_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        all_vecs.append(sub_vecs)
        if len(texts) > chunk_size:
            gc.collect()

    vecs = np.vstack(all_vecs).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def build_indices(chunk_sets: Dict[str, List[Dict]], store_dir: str) -> None:
    os.makedirs(store_dir, exist_ok=True)
    for strategy, chunks in chunk_sets.items():
        if not chunks:
            continue
        print(f"  Embedding {len(chunks)} chunks for strategy '{strategy}' (low memory mode)...", flush=True)
        texts = [c["text"] for c in chunks]
        vecs = embed_texts(texts)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        faiss.write_index(index, os.path.join(store_dir, f"{strategy}.faiss"))
        with open(os.path.join(store_dir, f"{strategy}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        del vecs, index, texts
        gc.collect()


def load_index(strategy: str, store_dir: str):
    index_path = os.path.join(store_dir, f"{strategy}.faiss")
    meta_path = os.path.join(store_dir, f"{strategy}_meta.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"Missing index for strategy '{strategy}' in {store_dir}. "
            f"Run data/prepare_dataset.py first."
        )
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
