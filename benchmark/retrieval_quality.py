"""
MSMARCO-XI marks one (or more) passage per query as `is_selected` (the gold
passage). We can use that as free ground truth to report actual retrieval
quality (Recall@k) alongside the latency numbers -- judges get a "does this
even work" signal, not just speed.

Usage:
    python benchmark/retrieval_quality.py --n-eval-queries 100 --k 5
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.retrieval import Retriever  # noqa: E402

STORE_DIR = os.environ.get("DATA_DIR", "./data/store")


def build_ground_truth(store_dir: str):
    """query_id -> {source_query, gold_doc_ids: set of doc_ids marked is_selected}"""
    meta_path = os.path.join(store_dir, "metadata_aware_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    by_query = defaultdict(lambda: {"source_query": None, "gold_doc_ids": set()})
    for c in chunks:
        qid = c["metadata"].get("query_id")
        if qid is None:
            continue
        by_query[qid]["source_query"] = c["metadata"].get("source_query")
        if c["metadata"].get("is_selected"):
            by_query[qid]["gold_doc_ids"].add(c["doc_id"])
    return by_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval-queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    ground_truth = build_ground_truth(STORE_DIR)
    evaluable = [
        (qid, v) for qid, v in ground_truth.items() if v["source_query"] and v["gold_doc_ids"]
    ][: args.n_eval_queries]

    if not evaluable:
        raise SystemExit(
            "No evaluable queries found (no is_selected passages in the indexed subset). "
            "Try a larger --n-queries when running prepare_dataset.py."
        )

    retriever = Retriever(STORE_DIR)
    hits = 0
    for qid, v in evaluable:
        retrieved, _ = retriever.retrieve(v["source_query"], top_n=args.k)
        # A "hit" = at least one retrieved doc_id's underlying source doc matches
        # a gold doc_id's passage index family (same query's passage list).
        retrieved_docs = {c.doc_id for c in retrieved}
        if retrieved_docs & v["gold_doc_ids"]:
            hits += 1

    recall_at_k = hits / len(evaluable)
    print(f"Evaluated {len(evaluable)} queries.")
    print(f"Recall@{args.k}: {recall_at_k:.3f} ({hits}/{len(evaluable)})")

    os.makedirs("benchmark", exist_ok=True)
    with open("benchmark/retrieval_quality.json", "w") as f:
        json.dump(
            {"n_eval_queries": len(evaluable), "k": args.k, "recall_at_k": recall_at_k}, f, indent=2
        )


if __name__ == "__main__":
    main()
