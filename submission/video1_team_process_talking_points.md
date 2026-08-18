# Video 1 — Team & Process (90 seconds)
## Talking Points (read from this while screen-recording)

**Goal of this video:** Show HOW we built it — workflow, decisions, debugging —
not the product itself. Judges want to see real engineering process.

---

## 📋 Pre-Recording Checklist (Staging)
- [ ] Close all unrelated browser tabs, messaging apps, and personal windows.
- [ ] Silence OS / mobile notifications and set "Do Not Disturb".
- [ ] Confirm `.env` file and API keys are **never** opened or visible on screen in IDE or terminal.
- [ ] Have the GitHub repo / IDE opened to `README.md` and key source files (`src/chunking.py`, `src/guardrails.py`, `src/orchestrator.py`).
- [ ] Test microphone audio levels in your screen recorder before recording.

---

### [0:00–0:15] — The Problem & Our Setup
- Task dropped Aug 13: build voice RAG on ai4bharat/MSMARCO-XI in 9 days.
- We used an agentic coding workflow from day one — AI pair programming handled
  boilerplate scaffolding (FastAPI server, FAISS index plumbing) so we could
  focus on the three decisions that actually matter: chunking strategy, guardrail
  design, latency budget.
- Divided ownership early: STT + server layer / chunking + retrieval / guardrails + harness.

### [0:15–0:35] — The Dataset Schema Debugging Session (this is the real story)
- MSMARCO-XI is NOT standard MS MARCO. We read the dataset card before writing
  a single line of data code — verified the actual field names:
  `query`, `passages.Translated_passages`, `passages.is_selected`.
  (Standard MS MARCO uses `query`, `passages.passage_text`, etc.)
- The schema is documented at the top of `data/prepare_dataset.py` as a warning
  because we almost got this wrong: one row = one query + multiple candidate
  passages, NOT one row per passage. We had to flatten it.
- Used HuggingFace streaming (`split="train", streaming=True`) to pull only the
  requested subset — avoids downloading all ~55 GB.

### [0:35–0:55] — Three Chunking Strategies: Why Not Just One
- Fixed-size chunking alone is naive — it cuts sentences mid-thought.
- We added semantic chunking (whole-sentence packing up to 80 words) and
  metadata-aware chunking (same split + provenance tags on every chunk).
- FAISS gets three separate indices; retrieval does a weighted merge (0.85/1.0/1.0)
  across all three and deduplicates by `doc_id`. This means we're not betting
  on one splitting philosophy — a real engineering decision, not just code quantity.

### [0:55–1:10] — Guardrails & Harness Design
- Three guardrail stages wired explicitly into the orchestrator:
  unsafe input check BEFORE any retrieval spend, off-topic check AFTER retrieval,
  groundedness check AFTER generation. This order matters — cheap checks first.
- The harness (`src/orchestrator.py`) wraps every stage: pydantic-typed results,
  tenacity retries on the LLM call, per-stage timing on every single request.
  Nothing raw ever surfaces to the caller.

### [1:10–1:25] — Latency Budget
- Interpreted the "under 200 ms" target as retrieval-side only — embed + search +
  guardrails. STT and Claude network calls are separately reported.
  This is industry-standard and we call it out explicitly in the benchmark output
  so judges don't think we're claiming sub-200 ms LLM generation.
- Benchmarked across 200 queries (not a single best-case run) for P50/P70/P100.

### [1:25–1:30] — Close
- Everything in one repo, reproducible setup (`python -m venv && pip install`),
  live deployment with zero code changes. That's the build.

---

*Keep it conversational — use these bullets as prompts, don't read word-for-word.*
