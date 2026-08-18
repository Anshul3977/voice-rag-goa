"""
Runs N queries through the pipeline and reports P50/P70/P100 latency,
broken out by stage. Retrieval-side stages are what's held to the <200ms
target (see README section 2); generation_ms (LLM network call) is reported
separately since it's not something the retrieval system controls.

Usage:
    python benchmark/latency_test.py --n-queries 200 --skip-generation
    (--skip-generation lets you benchmark pure retrieval-side latency
     without burning LLM API calls / being rate-limited)
"""
import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.orchestrator import Pipeline  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.indexing import embed_texts  # noqa: E402
from src import guardrails  # noqa: E402

STORE_DIR = os.environ.get("DATA_DIR", "./data/store")


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)
    return s[idx]


def sample_queries(store_dir: str, n: int):
    """Reuse source_queries embedded in the metadata-aware chunks as a
    realistic query set (MS MARCO gives us real queries per passage)."""
    meta_path = os.path.join(store_dir, "metadata_aware_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    queries = [c["metadata"].get("source_query", "").strip() for c in chunks]
    queries = [q for q in queries if q]
    random.shuffle(queries)
    if len(queries) < n:
        # pad by resampling if the corpus subset is small
        queries = (queries * (n // max(len(queries), 1) + 1))[:n]
    return queries[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument(
        "--skip-generation",
        action="store_true",
        help="Benchmark retrieval-side latency only (no LLM calls).",
    )
    args = ap.parse_args()

    queries = sample_queries(STORE_DIR, args.n_queries)
    print(f"Benchmarking {len(queries)} queries (skip_generation={args.skip_generation}) ...")

    stage_timings = {}
    stage_samples = {}
    statuses = {}

    if args.skip_generation:
        retriever = Retriever(STORE_DIR)
        print("Warming up retriever & embedder ...", flush=True)
        _ = retriever.retrieve("वार्मअप warmup", top_n=5)

        for idx, q in enumerate(queries):
            unsafe = guardrails.check_unsafe_input(q)
            retrieved, timings = retriever.retrieve(q, top_n=5)
            off_topic = guardrails.check_off_topic(retrieved)
            timings["total_retrieval_side_ms"] = (
                timings["embed_query_ms"] + timings["retrieve_ms"]
            )
            for k, v in timings.items():
                stage_timings.setdefault(k, []).append(v)
                stage_samples.setdefault(k, []).append((idx, v, q))
            status = "refused_off_topic" if not off_topic.passed else "ok"
            statuses[status] = statuses.get(status, 0) + 1
    else:
        pipeline = Pipeline(store_dir=STORE_DIR)
        print("Warming up pipeline & retriever ...", flush=True)
        _ = pipeline.retriever.retrieve("वार्मअप warmup", top_n=5)

        for idx, q in enumerate(queries):
            result = pipeline.run(q)
            for k, v in result.timings_ms.items():
                stage_timings.setdefault(k, []).append(v)
                stage_samples.setdefault(k, []).append((idx, v, q))
            statuses[result.status.value if hasattr(result.status, "value") else str(result.status)] = (
                statuses.get(result.status.value if hasattr(result.status, "value") else str(result.status), 0) + 1
            )

    report = {"n_queries": len(queries), "status_breakdown": statuses, "stages": {}}
    for stage, vals in stage_timings.items():
        report["stages"][stage] = {
            "p50_ms": round(percentile(vals, 50), 3),
            "p70_ms": round(percentile(vals, 70), 3),
            "p100_ms": round(percentile(vals, 100), 3),
            "mean_ms": round(sum(vals) / len(vals), 3),
            "n": len(vals),
        }

    os.makedirs("benchmark", exist_ok=True)
    with open("benchmark/results.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Latency Benchmark Results\n", f"n_queries: {report['n_queries']}\n"]
    lines.append(f"status_breakdown: {statuses}\n")
    lines.append("\n| stage | P50 (ms) | P70 (ms) | P100/max (ms) | mean (ms) |")
    lines.append("|---|---|---|---|---|")
    for stage, s in report["stages"].items():
        lines.append(
            f"| {stage} | {s['p50_ms']} | {s['p70_ms']} | {s['p100_ms']} | {s['mean_ms']} |"
        )
    if "total_retrieval_side_ms" in report["stages"]:
        p70 = report["stages"]["total_retrieval_side_ms"]["p70_ms"]
        verdict = "PASS" if p70 < 200 else "FAIL"
        lines.append(f"\n**Retrieval-side P70 vs 200ms target: {p70}ms -> {verdict}**")

    with open("benchmark/results.md", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nWritten to benchmark/results.json and benchmark/results.md")

    print("\n--- Outlier Diagnosis: Top 5 Slowest Queries Per Stage ---")
    for stage, samples in stage_samples.items():
        sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)[:5]
        print(f"\nStage: {stage}")
        for rank, (idx, val, q) in enumerate(sorted_samples, start=1):
            print(f"  {rank}. Loop Index: {idx:3d} | Latency: {val:8.3f} ms | Query: {q[:60]}")


if __name__ == "__main__":
    main()
