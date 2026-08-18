# Latency Benchmark Results

n_queries: 30

status_breakdown: {'error': 30}


| stage | P50 (ms) | P70 (ms) | P100/max (ms) | mean (ms) |
|---|---|---|---|---|
| guardrail_unsafe_ms | 0.024 | 0.034 | 0.659 | 0.064 |
| embed_query_ms | 32.13 | 47.449 | 104.671 | 42.071 |
| retrieve_ms | 2.768 | 5.378 | 102.591 | 9.233 |
| guardrail_off_topic_ms | 0.005 | 0.006 | 0.064 | 0.009 |
| total_retrieval_side_ms | 38.722 | 52.396 | 207.386 | 51.394 |
| generation_ms | 505.649 | 505.773 | 523.214 | 506.09 |

**Retrieval-side P70 vs 200ms target: 52.396ms -> PASS**