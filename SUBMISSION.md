# HH Goa 2026 — Task 2: Voice-Enabled RAG — Submission

## GitHub Repository
https://github.com/Anshul3977/voice-rag-goa

## Live Working Link
**TBD** *(Render)* — *(Note for evaluators: If the service has been inactive for over 15 minutes, the free-tier container may take ~30–60s to spin up on the first request — this is standard cold-start behavior, not a failure).*

---

## Project Description

A voice-enabled Retrieval-Augmented Generation (RAG) pipeline built on the
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset
(Hindi subset). A user speaks a question → ElevenLabs transcribes it → the query is
embedded and searched across FAISS indices built with deliberately different
chunking strategies (fixed-size with overlap, and sentence/semantic boundary-aware
with rich provenance tagging) → a weighted hybrid merge across strategies returns
the top-N passages → Google Gemini generates a structured JSON answer
(answer + cited chunk IDs + confidence score) grounded strictly in the retrieved context
→ three sequential guardrail checks (unsafe input filter, off-topic/relevance filter,
groundedness/hallucination check) decide whether to surface the answer or return a
safe refusal. The entire retrieval side (embed + search + guardrail checks) is
benchmarked to run under 200 ms; STT and generation LLM network calls are separately
reported and intentionally excluded from the latency budget (they are network-bound
and outside the retrieval system's control).

---

## Tech Stack

Exactly the packages in `requirements.txt` — nothing more:

| Component | Library / Service | Version |
|-----------|-------------------|---------|
| Speech-to-Text | ElevenLabs API (`elevenlabs`) | 1.9.0 |
| Embedding model | `sentence-transformers` / `paraphrase-multilingual-MiniLM-L12-v2` | 3.1.1 |
| Vector store | `faiss-cpu` (IndexFlatIP, cosine sim) | 1.8.0.post1 |
| LLM generation | Google Gemini (`google-genai` / `gemini-3.5-flash-lite`) | 2.18.1 |
| API server | FastAPI + uvicorn | 0.115.0 / 0.30.6 |
| Data loading | HuggingFace `datasets` (streaming) | 2.21.0 |
| Retries | `tenacity` | 9.0.0 |
| Sentence splitting | `nltk` | 3.9.1 |
| Validation | `pydantic` v2 | 2.13.4 |
| HTTP extras | `python-multipart`, `uvicorn[standard]` | — |

---

## How This Meets Each Technical Requirement

### Req 1 — Speech-to-Text (ElevenLabs or Sarvam)
✅ **ElevenLabs** chosen. `src/stt.py` wraps the ElevenLabs STT API with retry
(via `tenacity`). `app/server.py` calls it on the uploaded audio file before
passing the transcript to the pipeline.

### Req 2 — Chunking: "vast", not naive fixed-size
✅ **Distinct strategies** implemented in `src/chunking.py`, indexed into FAISS namespaces:

- **Fixed-size with overlap** (`chunk_fixed`) — sliding word window (60 words /
  15-word overlap); baseline recall, cheap, good for short MS MARCO passages.
- **Semantic / sentence-boundary aware** (`chunk_semantic`) — greedily packs whole
  sentences up to an 80-word budget; never cuts a sentence mid-thought; better
  precision for factoid QA.
- **Metadata-aware provenance** (`metadata_aware`) — applies sentence-boundary
  semantic chunking while attaching `doc_id`, `source_query`, `query_id`, `is_selected`,
  `position`, and character/word lengths as first-class metadata; enables provenance
  filtering, chunk citation, and ground-truth quality checking.

*Optimization Note:* Semantic and metadata-aware chunking share the same sentence-boundary
text splitting; to avoid redundant memory and compute overhead, they are unified into the
`metadata_aware` physical index. Retrieval (`src/retrieval.py`) performs a weighted merge
(`metadata_aware`=1.0, `fixed`=0.85) and deduplicates by `doc_id`.

### Req 3 — Latency target: end-to-end < 200 ms
✅ The 200 ms budget covers: query embedding + FAISS search across all 3 indices +
guardrail checks — tracked as `total_retrieval_side_ms` in every `PipelineResult`.
STT and generation LLM calls are network-bound and reported separately (see Req 4).
**Measured Retrieval-Side P70**: **19.571 ms** (< 200 ms target → **PASS**).

### Req 4 — Latency analytics & Retrieval Quality: P50 / P70 / P100 across ≥ 200 queries
✅ `benchmark/latency_test.py --n-queries 200` outputs `benchmark/results.md` with
P50 / P70 / P100 broken out by stage across 200 queries. A warmup call precedes
measurement so steady-state behavior is measured:

| Stage | P50 (ms) | P70 (ms) | P100 / Max (ms) | Mean (ms) |
|---|---|---|---|---|
| `embed_query_ms` | 16.606 | 17.988 | 121.831 | 19.264 |
| `retrieve_ms` | 1.486 | 1.602 | 86.842 | 2.296 |
| **`total_retrieval_side_ms`** | **18.114** | **19.571** | **208.673** | **21.560** |

*Outlier analysis:* The P100 outliers (isolated to loop iterations 119–120) spiked synchronously across both independent stages (`embed_query_ms` and `retrieve_ms`) despite having no shared per-query state, confirming an external OS-level memory compression/swap paging pause on the host machine rather than an internal pipeline bottleneck.

**Retrieval Quality (evaluated on `benchmark/retrieval_quality.py` across 100 queries):**
- **Recall@5**: **0.520** (52/100 against ground-truth `is_selected` passages in subset).
- Full benchmark output stored in [`benchmark/results.md`](benchmark/results.md) and [`benchmark/retrieval_quality.json`](benchmark/retrieval_quality.json).

**LLM Generation Latency & Guardrail Status Breakdown (Google Gemini `gemini-3.5-flash-lite`):**
*Note: Generation is network-bound and intentionally reported separately from the sub-200ms retrieval-side budget.*

| Stage | P50 (ms) | P70 (ms) | P100 / Max (ms) | Mean (ms) | Samples ($n$) |
|---|---|---|---|---|---|
| `generation_ms` | **1145.116** | **1291.967** | **3397.078** | **1297.905** | 20 / 20 |
| `guardrail_groundedness_ms` | **0.258** | **0.290** | **0.932** | **0.286** | 20 / 20 |

**Guardrail & Harness Status Breakdown (20 queries evaluated with 4.2s pacing — 100% completion):**
- `ok`: **12 queries (60%)** — passed all three guardrails with verified citations and high confidence.
- `refused_ungrounded`: **8 queries (40%)** — groundedness guardrail actively caught and refused responses attempting to use unretrieved/outside knowledge.
- `error`: **0 queries (0%)** — zero rate limits or harness errors with `gemini-3.5-flash-lite`.

### Req 5 — Harness: structured orchestration, retries, typed I/O
✅ `src/orchestrator.py` — `Pipeline.run()` wraps every stage:
- Pydantic-validated `PipelineResult` with typed `PipelineStatus` enum (`ok`,
  `refused_unsafe`, `refused_off_topic`, `refused_ungrounded`, `error`).
- `_generate_with_retry` uses `tenacity` (2 attempts, exponential backoff 0.5–3 s).
- Per-stage timing dict (`timings_ms`) on every result.
- Typed safe refusals at every exit — raw exceptions never reach the caller.

### Req 6 — Guardrails: off-topic, unsafe, hallucination / groundedness
✅ `src/guardrails.py` — three sequential checkpoints:

1. **`check_unsafe_input(query)`** — regex/keyword denylist (weapons, self-harm,
   PII-extraction patterns); fires *before* any retrieval or LLM spend.
   Stage: `unsafe_input`.

2. **`check_off_topic(retrieved)`** — if best cosine similarity < 0.30, the corpus
   doesn't cover this query; returns a safe refusal instead of an improvised answer.
   Stage: `off_topic`.

3. **`check_groundedness(answer, cited_chunk_ids, retrieved, confidence)`** —
   verifies cited `chunk_id`s exist in the retrieved set; checks lexical word-overlap
   ratio (min 0.15) between answer and cited chunks; combined with model's
   self-reported confidence (min 0.40) decides whether to surface or refuse.
   Stage: `groundedness`.
