# Latency Benchmark Results

Retrieval is BM25 over the committed chunk files. Generation is omitted when `--skip-generation` is set.

n_queries: 20

status_breakdown: {'ok': 19, 'refused_off_topic': 1}


| stage | P50 (ms) | P70 (ms) | P100/max (ms) | mean (ms) |
|---|---|---|---|---|
| embed_query_ms | 0.003 | 0.003 | 0.006 | 0.003 |
| retrieve_ms | 0.706 | 1.379 | 3.903 | 1.076 |
| total_retrieval_side_ms | 0.71 | 1.381 | 3.908 | 1.079 |

**Retrieval-side P70 vs 200ms target: 1.381ms -> PASS**