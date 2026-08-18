---
title: Voice RAG Goa
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Voice-Enabled RAG — HH Goa 2026, Task 2

Pipeline: **Voice input → Speech-to-Text (ElevenLabs) → Multi-strategy chunking/retrieval (FAISS) → Grounded answer generation**, wrapped in a retry/error-handling harness with guardrails, and benchmarked for P50/P70/P100 latency.

Dataset: [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)

---

## 1. Architecture

```
audio file/mic ──▶ STT (ElevenLabs)  ──▶ query text
                                            │
                                            ▼
                                 ┌─ Orchestrator/Harness ─┐
                                 │  Guardrail: input check │
                                 │  Retrieval (hybrid)     │
                                 │  Guardrail: relevance   │
                                 │  Generation (LLM)       │
                                 │  Guardrail: groundedness│
                                 └──────────────────────────┘
                                            │
                                            ▼
                                     answer + citations
```

### Why this design
- **STT**: ElevenLabs `speech_to_text` API — chosen for good multilingual support (dataset is Indian-language / MS MARCO translated), low-latency streaming option.
- **Chunking is deliberately NOT a single fixed-size splitter.** `src/chunking.py` implements three strategies and a router:
  1. **Fixed-size with overlap** — baseline, cheap, good recall for short passages.
  2. **Sentence/semantic chunking** — splits on sentence boundaries then greedily packs sentences up to a token budget, preserving semantic coherence (better precision for factoid QA like MS MARCO).
  3. **Metadata-aware chunking** — retains `doc_id`, `passage_id`, `source_query` (MS MARCO has query-passage pairs) as chunk metadata so retrieval can filter/boost by metadata and answers can cite provenance.
  - All three are indexed into **separate FAISS namespaces**, and retrieval does a **weighted merge** across them (`src/retrieval.py`), so we're not committed to one splitting philosophy.
- **Vector DB**: FAISS (`IndexFlatIP` on normalized embeddings = cosine similarity), in-memory, sub-millisecond search for this dataset size. Embeddings via `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim, CPU-friendly — critical for the 200ms budget).
- **Generation**: LLM call (Anthropic Claude by default, swappable) with a strict "answer only from context" system prompt, forced to output structured JSON (answer, used_chunk_ids, confidence).
- **Harness** (`src/orchestrator.py`): pydantic-validated I/O at every stage, retry with exponential backoff on transient failures, typed error results instead of raw exceptions bubbling to the caller, and a single `PipelineResult` object logged with per-stage timings.
- **Guardrails** (`src/guardrails.py`):
  - *Off-topic filter*: embedding-similarity check between query and the whole corpus centroid + keyword/topic classifier; queries below threshold get a polite "out of scope" response instead of a hallucinated answer.
  - *Unsafe input filter*: lightweight regex + keyword denylist for self-harm/violence/PII-extraction style prompts → refused before hitting retrieval or the LLM.
  - *Groundedness / hallucination check*: after generation, we check that every claim's cited `chunk_id` actually appears in the retrieved set, and run a lexical-overlap ("does the answer's content actually appear in the cited chunk") sanity check. Low overlap + low LLM-reported confidence ⇒ system responds "I don't have enough information in the provided context" instead of guessing.

## 2. Latency budget (<200ms end-to-end, excluding STT + LLM network round-trip)

The 200ms target is interpreted as **chunking + embedding of the query + vector search + guardrail checks** (the "retrieval-side" of the pipeline) — this is standard practice since STT and generation LLM calls are network-bound (typically 500ms–2s) and outside the retrieval system's control. This is stated explicitly in `benchmark/latency_test.py` output and should be called out in your demo video so judges don't think you're claiming sub-200ms LLM generation.

Run:
```bash
python benchmark/latency_test.py --n-queries 200
```
Outputs `benchmark/results.json` and `benchmark/results.md` with P50 / P70 / P100 (max) broken out by stage: `embed_query_ms`, `retrieve_ms`, `guardrail_ms`, `total_retrieval_side_ms`, and (separately, clearly labeled) `generation_ms`.

Retrieval **quality** (not just speed) is also measurable for free: MSMARCO-XI marks a gold passage per query (`is_selected`), so `benchmark/retrieval_quality.py` reports Recall@k against real ground truth — worth including in your submission alongside the latency table.

## 3. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ELEVENLABS_API_KEY and ANTHROPIC_API_KEY (or OPENAI_API_KEY)

python data/prepare_dataset.py --n-queries 800 --lang hi   # downloads + subsets MSMARCO-XI, builds indices
python benchmark/latency_test.py --n-queries 200            # latency report (P50/P70/P100)
python benchmark/retrieval_quality.py --n-eval-queries 100  # Recall@k against MSMARCO's is_selected ground truth
uvicorn app.server:app --reload                               # run the live app on :8000
```

Open `app/static/index.html` (served at `/`) — record a question, get a spoken-context-grounded answer back.

## 4. Repo map
```
data/prepare_dataset.py   dataset download, cleaning, chunk-and-index build (all 3 strategies)
src/stt.py                 ElevenLabs STT wrapper w/ retry
src/chunking.py             fixed / semantic / metadata-aware chunkers
src/indexing.py               FAISS index build + persistence per strategy
src/retrieval.py               hybrid weighted-merge retriever
src/guardrails.py                off-topic / unsafe / groundedness checks
src/generation.py                 LLM answer generation, structured JSON output
src/orchestrator.py               harness: retries, typed errors, stage timing
src/pipeline.py                    end-to-end entrypoint used by app + benchmark
benchmark/latency_test.py           P50/P70/P100 harness
app/server.py                        FastAPI: POST /ask (audio) -> answer JSON
app/static/index.html                 minimal recorder UI
tests/                                 unit tests for chunking + guardrails
```

## 5. Deployment (for the "Live working link" requirement)
Any of these work with zero code changes:
- **Render / Railway / Fly.io**: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`, add env vars in dashboard.
- **HuggingFace Spaces (Docker SDK)**: add the included `Dockerfile`, push repo, set secrets in Space settings.
- Free tier is enough — the FAISS index is in-memory and small (a few thousand passages).

## 6. Submission checklist (don't lose points on process, not code)
- [ ] GitHub repo public, this README updated with your live link
- [ ] `.env` / API keys **not committed** (check `.gitignore`)
- [ ] `benchmark/results.md` committed with real P50/P70/P100 numbers from ≥200 queries
- [ ] Live link tested in an incognito window right before submitting
- [ ] Video 1 (90s, process/team) — screen-record a standup/whiteboard/pairing session, not the app
- [ ] Video 2 (demo, end-to-end voice question → spoken/text answer, show a guardrail refusal too)
- [ ] Both videos posted on **Instagram, X, LinkedIn** — **by every team member individually**
- [ ] Every single post includes `#RAGInGoa`
- [ ] At least one Instagram account among the team is public
- [ ] Google Form submitted — no resubmissions allowed, so submit only the final commit hash
