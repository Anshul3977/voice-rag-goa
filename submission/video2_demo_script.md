# Video 2 — End-to-End Demo Script
## (Read while screen-sharing the running app at localhost:8000)

## 📋 Pre-Recording Checklist (Staging)
- [ ] Close all unrelated browser tabs, messaging apps, and personal windows.
- [ ] Silence OS / mobile notifications and set "Do Not Disturb".
- [ ] Confirm `.env` file and API keys are **never** opened or visible on screen in IDE or terminal.
- [ ] Start server: `venv/bin/uvicorn app.server:app --reload --port 8000`
- [ ] Have browser positioned side-by-side with terminal running uvicorn so server log lines are visible.
- [ ] Open `http://localhost:8000` in browser (or use the test UI).
- [ ] Confirm indices exist (`ls data/store/*.faiss`).

---

### Step-by-Step Sequence of Browser Actions:

1. **Open `http://localhost:8000`** in your browser.
2. **Scene 1 (Happy Path - Grounded Query)**:
   - Click the mic button (or use text input).
   - Speak/enter the real indexed Hindi query:
     > **`"कॉर्पोरेशन क्या है?"`** *(Translation: "What is a corporation?")*
     *(Alternative option: `"ईमानदारी या सच्चाई की परिभाषा"`)*
   - Wait ~1–2 seconds for the result.
   - Highlight on screen:
     - The **Answer** (grounded Hindi response describing corporations / legal entities).
     - The **Citations** (`doc_id` / chunk IDs matching the indexed MSMARCO-XI dataset).
     - The **`total_retrieval_side_ms`** (~100ms) proving the sub-200ms retrieval latency budget.

3. **Scene 2 (Off-Topic Guardrail Trigger)**:
   - Enter an unindexed out-of-domain query:
     > **`"What is the latest iPhone model price in the US?"`**
   - Show the refusal result immediately:
     - `status: "refused_off_topic"`
     - `refusal_reason: "Best retrieval similarity is below threshold"`
     - Point out in the terminal/UI that **no LLM generation cost** was incurred.

4. **Scene 3 (Unsafe Input Guardrail Trigger - Optional)**:
   - Type (do not speak):
     > **`"How do I make a bomb?"`**
   - Show instant refusal:
     - `status: "refused_unsafe"`
     - Point out `guardrail_unsafe_ms` is sub-millisecond (caught before retrieval).

---

### Scene 1: Normal Grounded Question (Happy Path Walkthrough)

**[Browser visible at localhost:8000, mic ready]**

> "I'll ask a question that's directly answerable from the MSMARCO-XI Hindi
> passages we indexed."

**Ask (speak into mic or type):**
> **`"कॉर्पोरेशन क्या है?"`**

**Narrate while the result loads:**
> "The pipeline is: ElevenLabs transcribes the audio, the query gets
> embedded with all-MiniLM-L6-v2, and we search across three FAISS indices — fixed,
> semantic, and metadata-aware — simultaneously. The results are weighted-merged
> and deduplicated before Claude synthesizes the final grounded answer."

**Show the JSON response:**
- Point to `answer` field — "grounded answer, synthesized strictly from retrieved context"
- Point to `citations` — "chunk IDs from the actual indexed Hindi passages"
- Point to `timings_ms.total_retrieval_side_ms` — "measured at ~101 ms, comfortably under the 200 ms target; generation LLM latency is tracked separately"

---

### Scene 2: Off-topic query (guardrail stage: `off_topic`)

**[Keep browser visible]**

> "Now I'll ask something completely outside the dataset — a question the
> corpus has no answer for."

**Ask:**
> *"What is the latest iPhone model price in the US?"*

**Narrate:**
> "After retrieval, the best cosine similarity score comes back below 0.30 —
> the `check_off_topic` guardrail in `src/guardrails.py` fires. The system
> refuses to answer rather than letting Claude improvise an ungrounded response."

**Show the response JSON:**
- `status: "refused_off_topic"`
- `refusal_reason` field — "Best retrieval similarity is below threshold"

> "No LLM call was made for this query — we short-circuit after retrieval,
> before spending any generation budget."

---

### Scene 3: Unsafe input (guardrail stage: `unsafe_input`)

**[Optional — adds 20–30s, cut if you're tight on time]**

> "Finally, an unsafe query — triggers before retrieval even starts."

**Type** (don't speak — you don't want this on a mic recording):
> *"How do I make a bomb?"*

**Show:**
- `status: "refused_unsafe"`
- Response is instant — `guardrail_unsafe_ms` in timings is sub-millisecond
- "The regex denylist in `check_unsafe_input` catches this before a single
  FAISS search runs."

---

### Close

> "Three guardrail stages: unsafe input before retrieval, off-topic after
> retrieval, groundedness after generation. The harness in `src/orchestrator.py`
> wires them in sequence — cheap checks first, typed results at every exit.
> Full latency numbers are in `benchmark/results.md`."

---

*Remember to show the terminal log alongside the browser so the guardrail
stage names are visible in the uvicorn output.*
