# 2. Structured output, schema design and guardrails

How to turn a free-form conversation into reliable, typed data with an 8B local model, and what to do when the model still breaks the rules.

**Where to look:** [`models/entry.py`](../../backend/src/diary_agent/models/entry.py), [`services/extraction.py`](../../backend/src/diary_agent/services/extraction.py), [`models/summary.py`](../../backend/src/diary_agent/models/summary.py), [`models/memory.py`](../../backend/src/diary_agent/models/memory.py)

---

### 1. `with_structured_output(..., include_raw=True)` makes failures visible
- **Concept:** `model.with_structured_output(Schema, method="json_schema", include_raw=True)` returns `{"raw", "parsed", "parsing_error"}` instead of raising on bad output.
- **In this repo:** `services/extraction.py → build_extractor()`.
- **Why it matters:** `include_raw` lets you see *why* parsing failed and feed that error back to the model (point 8). Without it, you only get an exception.

### 2. Grammar-constrained decoding guarantees structure, not content
- **Concept:** Ollama compiles the JSON schema into a grammar that limits which tokens the model may pick next. The output is always valid JSON in the right shape, but the *values* can still be wrong.
- **In this repo:** Every extraction goes through this: entries, summaries, memory candidates.
- **Why it matters:** "It parsed" is not "it's correct". Validity is the grammar's job; correctness needs guardrails and evals.

### 3. Make every field required
- **Concept:** The grammar only forces fields that are *required*. Fields with defaults are optional, and a small model will often skip them.
- **In this repo:** Every field in `ExtractedEntry`, `PeriodSummary` and `MemoryItem` is required. "Nothing" is expressed as `[]` or `null`.
- **Why it matters (bug):** With defaults, `mood` came back `null` in 4 of 6 cases, and `tags`/`mood_words` were always empty. The raw output was just `{"summary": ..., "tags": ...}`.

### 4. Put a bound on every list
- **Concept:** An unbounded list lets the model keep generating forever.
- **In this repo:** `max_length` on every list field (e.g. `MemoryCandidates.memories` has `max_length=5`), plus a `num_predict=512` output cap as a backstop.
- **Why it matters (bug):** The extraction probe hung for more than 10 minutes at 98% GPU. The model was repeating tags in a loop: `"surgery", "went", "well", "surgery", …`.

### 5. The model never sees your field descriptions, so put them in the prompt
- **Concept:** The grammar uses the schema's *structure*. The `Field(description=...)` text doesn't reach the model.
- **In this repo:** `services/extraction.py → field_guide(schema)` renders each field's description into the system prompt. It's reused for summaries and memories.
- **Why it matters:** Descriptions seemed to have no effect until we noticed this. Your carefully written guidance does nothing unless the model can actually read it.

### 6. Field order matters: evidence before judgement
- **Concept:** The model writes JSON top to bottom. Fields it writes first become context for the fields that come after.
- **In this repo:** `ExtractedEntry` puts highlights and challenges before `mood`. `MemoryItem` puts `evidence` (the user's exact quote) before `content`.
- **Why it matters:** Writing the evidence first grounds the judgement. Having the quote also gives code something to check (see `apply_memory_guardrails` in [note 7](07-memory.md)).

### 7. Two-layer schemas: never ask the model for what code already knows
- **Concept:** Split "what the LLM infers" from "what the code knows". The LLM fills the first; code wraps it with the rest.
- **In this repo:** `ExtractedEntry` (LLM) → `JournalEntry` (+ date, transcript, model, prompt version). The same pattern appears as `PeriodSummary` → `SummaryRecord` (+ date range, entry count, `mood_avg`) and `MemoryItem` → `Memory` (+ id, source date, timestamps).
- **Why it matters:** Every field you ask the model for is a field it can get wrong. The date, the average mood and the model name are free and exact in code.

### 8. Retry with feedback
- **Concept:** When parsing fails, send the error back to the model and ask again, a bounded number of times.
- **In this repo:** `services/extraction.py → invoke_structured()` (shared by entries, summaries and memory extraction, `max_retries=2`).
- **Why it matters:** Small models often fix their own mistake once they're told what it is. Keep the retry loop in one place so every structured call behaves the same.

### 9. Enforce checkable rules in code, not in the prompt
- **Concept:** If a rule can be checked with plain code, enforce it after the model runs. It's 100% reliable, can be unit-tested, and can't be broken by a later prompt change.
- **In this repo:** `services/extraction.py → apply_guardrails()` drops gratitude the user never expressed, and mood words the user didn't use themselves. The summary and memory services each have their own `apply_guardrails` / `apply_memory_guardrails`.
- **Why it matters:** A prompt fix (v1.4) fixed one case but **broke another**. The guardrail fixed both, permanently. The trade-off is that it drops some legitimate inferences ("happy" from "great day"). That's acceptable here because the spec says "use the user's own words".

---

### Try it
- Remove `max_length` from `tags` in `models/entry.py` and run `uv run python evals/run_evals.py --only extraction` (be ready to Ctrl+C).
- Print `field_guide()` to see exactly what the model is told about each field.
- Write a sarcastic transcript ("Oh *great*, another Monday.") into `evals/datasets/extraction.jsonl`. Does the mood come out right?

**Full story:** [NOTES.md](../NOTES.md) → M2 ("How Ollama structured output actually works", "The eval loop in practice").
