# 6. Context budgets and map-reduce summaries

How to summarise a week or a month of entries with a model that has a small context window, and why the best LLM call is often the one you don't make.

**Where to look:** [`services/summary.py`](../../backend/src/diary_agent/services/summary.py), [`models/summary.py`](../../backend/src/diary_agent/models/summary.py), [`repositories/summary_repository.py`](../../backend/src/diary_agent/repositories/summary_repository.py), [`cli/app.py`](../../backend/src/diary_agent/cli/app.py) (`summary`), [`evals/summary_probe.py`](../../evals/summary_probe.py)

---

### 1. The context window is a hard budget, and overflow fails silently
- **Concept:** A model can only read a fixed number of tokens. Ollama's default window is a few thousand tokens, and anything past it is **cut without any error**.
- **In this repo:** Every summary call sets `NUM_CTX = 8192` (`get_summary_model`). They all use the *same* value, because changing `num_ctx` between requests makes Ollama reload the model.
- **Why it matters:** With a small window, the model simply never sees the start of the month, and nothing tells you. Set the window explicitly and know how much you're sending.

### 2. Estimate tokens cheaply, and know the estimate's limits
- **Concept:** You don't need an exact tokenizer to budget; a rough estimate works.
- **In this repo:** `estimate_tokens()` uses characters / 4. `REDUCE_INPUT_TOKENS` is the budget for the reduce step's input.
- **Why it matters:** It's fast and good enough for English. It underestimates for CJK text, which is a known limit.

### 3. Map-reduce is a budgeting tool
- **Concept:** **Map** turns each item into something small; **collapse** merges groups only if the result is still too big; **reduce** writes the final answer from the small pieces. Every call has a known, bounded input.
- **In this repo:** `summarize_period()`: map = `day_line(entry)` per day → `collapse()` only if over `REDUCE_INPUT_TOKENS` → `reduce()` writes a structured `PeriodSummary`.
- **Why it matters:** It's how you fit unbounded input into a bounded window. But it's not a *quality* tool: every extra LLM hop is another chance to distort (point 8).

### 4. Don't redo a step you already have
- **Concept:** Before adding an LLM step, check whether an earlier step already produced what you need.
- **In this repo:** The map step was originally an LLM call per entry that condensed each transcript. It was replaced with `day_line()`, which builds one line per day **in code** from fields extraction already wrote *and the user approved*: summary, highlights, challenges, gratitude, mood words.
- **Why it matters (measured):** Every check still passed, sparse-week wording improved from 2/3 to 3/3, and a month review went from ~15 s to **4.3 s** (1 LLM call instead of 11). The old LLM map was a second, *unreviewed* copy of the day, and it was where a distortion came from (point 8). A per-entry stored "digest" was also tried and dropped: verbatim fragments like "No, skipped it." lose their context.

### 5. Collapse only when you must, because merged text has a floor
- **Concept:** Merging summaries of summaries loses detail, and text can only get so short before it's useless.
- **In this repo:** `collapse()` runs only when the input is over budget, for at most `MAX_COLLAPSE_ROUNDS`. A 31-day month of day lines is ~2,000 tokens, so it normally fits without collapsing.
- **Why it matters (experiment):** With the budget forced down to 150 tokens, the month needed 9 extra calls, took 2.3× longer, was *still* over budget after 3 rounds, and lost the per-day dates.

### 6. Code computes the numbers
- **Concept:** LLMs are bad at arithmetic and counting. Compute statistics in code and give them to the model as facts.
- **In this repo:** `stats_text()` computes the mood average, the lowest and highest days, and tags that recur on 2+ days. `mood_avg` is stored on `SummaryRecord` by code. If there are no entries, there's no LLM call at all.
- **Why it matters:** Numbers from code are exact and free. The model's job is to *interpret* them.

### 7. …but more facts in the prompt aren't automatically better
- **Concept:** The model talks about whatever you give it. Each new input changes what it focuses on, so measure each addition.
- **In this repo:** A computed "Mood trend: down (6.8 → 6.0)" line was added to the prompt, then **removed** (the reason is in the `stats_text` docstring).
- **Why it matters (eval):** With the trend line, the review named both the low and high day in only **2/6** runs. Without it, **6/6**. The model narrated the direction and dropped the specifics.

### 8. Errors compound across hops, so inspect the intermediate outputs
- **Concept:** In a multi-step pipeline, a mistake early on gets repeated and polished by later steps.
- **In this repo:** A short entry, "Your first project is now complete", was padded by the old map step into "completed pottery class", and the reduce step repeated it. The cause: the map prompt asked for "2-3 sentences" about a one-line entry. **Length demands cause invention.**
- **Why it matters:** The final review looked fluent. Only looking at the intermediate digests (the probe saves them) showed where the error came from.

### 9. Where a rule is checkable, the code writes the answer
- **Concept:** When the model keeps breaking a rule you can check in code, let code override that field.
- **In this repo:** `services/summary.py → apply_guardrails()`: with fewer than `MIN_TREND_DAYS` (3) rated days, code writes `mood_trend` itself. It also drops "good mood" listed as a win or challenge.
- **Why it matters:** "Remained steady at 7/10 throughout the period" for 2 days out of 7. The prompt rule against this scored **0/3**; the code override fixed it. Two data points aren't a trend.

---

### Try it
- `uv run python evals/summary_probe.py --runs 3`, then `--collapse`, and compare the reduce prompts saved in `evals/results/`.
- Print `day_line(e)` for a few entries: that's exactly what the reduce step sees for each day.
- Set `num_ctx=2048` in `get_summary_model` and add a very long entry: watch the start of the month disappear without an error.

**Full story:** [NOTES.md](../NOTES.md) → M5 ("What the probe found", "Later change: the map step is now code").
