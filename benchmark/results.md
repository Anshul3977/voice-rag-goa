# Latency Benchmark Results

n_queries: 20

status_breakdown: {'ok': 12, 'refused_ungrounded': 8}


| stage | P50 (ms) | P70 (ms) | P100/max (ms) | mean (ms) |
|---|---|---|---|---|
| guardrail_unsafe_ms | 0.148 | 0.202 | 0.379 | 0.163 |
| embed_query_ms | 102.29 | 143.891 | 265.989 | 130.354 |
| retrieve_ms | 4.723 | 12.073 | 27.213 | 7.96 |
| guardrail_off_topic_ms | 0.01 | 0.016 | 0.085 | 0.017 |
| total_retrieval_side_ms | 116.262 | 150.007 | 280.189 | 138.535 |
| generation_ms | 1145.116 | 1291.967 | 3397.078 | 1297.905 |
| guardrail_groundedness_ms | 0.258 | 0.29 | 0.932 | 0.286 |

**Retrieval-side P70 vs 200ms target: 150.007ms -> PASS**