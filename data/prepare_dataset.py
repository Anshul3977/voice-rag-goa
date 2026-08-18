"""
Downloads ai4bharat/MSMARCO-XI, subsets it to a manageable number of passages
for a hackathon-scale FAISS index, and hands off to src/indexing.py to build
all three chunking-strategy indices.

ACTUAL schema (verified against the HF dataset card and loading script):

  The dataset has ONE default config (no per-language configs like 'hi', 'bn').
  Language is NOT selected via a builder config arg; instead every row carries
  source_lang / target_lang fields using FLORES-200 codes (e.g. "eng_Latn",
  "asm_Beng", "hin_Deva"). We filter the stream post-hoc by target_lang field.

  row = {
    "query": str,                # translated query text
    "Answer": str,                # translated answer text
    "query_id": int,
    "query_type": str,             # e.g. "DESCRIPTION"
    "source_lang": str,            # always "eng_Latn" (source is English)
    "target_lang": str,            # FLORES-200 code, e.g. "hin_Deva" for Hindi
    "passages": {
        "is_selected": [0/1, ...],  # 1 = this passage is the gold/relevant one
        "English_passages": [str, ...],
        "Translated_passages": [str, ...],
    },
    "Eng_Query": str, "Eng_Answer": str,
    "meta": dict,                  # translation model metadata
  }
One row = one query with MULTIPLE candidate passages, not one row per passage.
We flatten: each (row, passage_index) pair becomes one retrievable chunk-source,
tagged with is_selected so retrieval quality can be checked against ground truth.

Usage:
    python data/prepare_dataset.py --n-queries 800 --lang hi
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from src.chunking import build_all_chunks  # noqa: E402
from src.indexing import build_indices  # noqa: E402

DEFAULT_STORE = os.path.join(os.path.dirname(__file__), "store")

# Two-letter shorthand codes kept for CLI compatibility (--lang hi).
VALID_LANGS = {
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "ur"
}

# Maps two-letter CLI --lang codes to the FLORES-200 prefix used in target_lang field
_LANG_PREFIX = {
    "as": "asm_", "bn": "ben_", "gu": "guj_", "hi": "hin_",
    "kn": "kan_", "ml": "mal_", "mr": "mar_", "ne": "npi_",
    "or": "ory_", "pa": "pan_", "sa": "san_", "ta": "tam_",
    "ur": "urd_",
}

_LANG_FILE = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin",
    "kn": "kan", "ml": "mal", "mr": "mar", "ne": "nep",
    "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
    "ur": "urd",
}


def load_passages(lang: str, n_queries: int, use_translated: bool = True):
    """
    Streams ai4bharat/MSMARCO-XI, filters rows post-hoc by target_lang prefix,
    and stops once n_queries matching rows are collected.
    """
    if lang not in VALID_LANGS:
        raise SystemExit(
            f"'{lang}' is not a recognised --lang code. Valid: {sorted(VALID_LANGS)}"
        )

    lang_prefix = _LANG_PREFIX[lang]
    file_stem = _LANG_FILE[lang]
    rel_path = f"validation/{file_stem}val.parquet"
    local_cached = os.path.expanduser(f"~/.cache/huggingface/downloads/{file_stem}val.parquet")

    if os.path.exists(local_cached) and os.path.getsize(local_cached) > 0:
        print(f"Loading ai4bharat/MSMARCO-XI from local cache ({local_cached}, filtering target_lang prefix='{lang_prefix}') ...", flush=True)
        local_file = local_cached
    else:
        print(f"Loading ai4bharat/MSMARCO-XI ({rel_path}, filtering target_lang prefix='{lang_prefix}') ...", flush=True)
        local_file = hf_hub_download("ai4bharat/MSMARCO-XI", filename=rel_path, repo_type="dataset")

    ds = load_dataset("parquet", data_files={"train": local_file}, split="train", streaming=True)

    passages = []
    matched = 0
    scanned = 0
    first_target_lang_seen = None

    for row in ds:
        scanned += 1
        target_lang = row.get("target_lang") or ""
        if not target_lang.startswith(lang_prefix):
            continue

        if first_target_lang_seen is None:
            first_target_lang_seen = target_lang
            print(f"  First matching row: target_lang='{target_lang}' (scanned {scanned} rows so far)", flush=True)

        matched += 1
        query = (row.get("query") or "").strip()
        qid = row.get("query_id", matched)
        p = row.get("passages") or {}
        texts = p.get("Translated_passages") if use_translated else p.get("English_passages")
        selected_flags = p.get("is_selected") or []
        if not texts:
            if matched >= n_queries:
                break
            continue

        for j, text in enumerate(texts):
            if not text or not text.strip():
                continue
            is_selected = bool(selected_flags[j]) if j < len(selected_flags) else False
            passages.append(
                {
                    "doc_id": f"{qid}_{j}",
                    "text": text.strip(),
                    "source_query": query,
                    "query_id": qid,
                    "is_selected": is_selected,
                }
            )

        if matched >= n_queries:
            break

    print(f"Scanned {scanned} total rows to find {matched} matches for '{lang}' (target_lang prefix '{lang_prefix}').", flush=True)
    print(f"Loaded {len(passages)} passages from {matched} queries.", flush=True)
    return passages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-queries", type=int, default=800,
        help="Number of MSMARCO-XI query rows to pull (each yields ~10 passages)",
    )
    ap.add_argument(
        "--lang", type=str, default="hi",
        help=(
            "Two-letter language code to index, e.g. hi/bn/ta. "
            "Filtering is performed post-hoc on target_lang field rather than selecting a builder config."
        ),
    )
    ap.add_argument(
        "--use-english", action="store_true",
        help="Index English_passages instead of Translated_passages",
    )
    ap.add_argument("--store", type=str, default=DEFAULT_STORE)
    args = ap.parse_args()

    os.makedirs(args.store, exist_ok=True)
    raw_path = os.path.join(args.store, "raw_passages.json")
    if os.path.exists(raw_path):
        print(f"Loading existing passages from {raw_path} ...", flush=True)
        with open(raw_path, "r", encoding="utf-8") as f:
            passages = json.load(f)
        print(f"Loaded {len(passages)} passages from cache.", flush=True)
    else:
        passages = load_passages(args.lang, args.n_queries, use_translated=not args.use_english)
        if not passages:
            raise SystemExit("No passages loaded — check dataset on HF hub page.")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False)

    print("Building chunks (fixed / semantic / metadata-aware) ...", flush=True)
    chunk_sets = build_all_chunks(passages)
    for strategy, chunks in chunk_sets.items():
        print(f"  {strategy}: {len(chunks)} chunks", flush=True)

    print("Building FAISS indices ...", flush=True)
    build_indices(chunk_sets, store_dir=args.store)
    print(f"Done. Indices + metadata written to {args.store}", flush=True)


if __name__ == "__main__":
    main()
