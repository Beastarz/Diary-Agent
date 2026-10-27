# 8. Evals: measuring an agent

Every important fix in this project came from a number, not a hunch. This note covers how those numbers were produced, and the ways they misled us.

**Where to look:** [`evals/run_evals.py`](../../evals/run_evals.py), [`evals/evaluators.py`](../../evals/evaluators.py), [`evals/suites/`](../../evals/suites/), [`evals/datasets/`](../../evals/datasets/), [`evals/tests/`](../../evals/tests/), the probes in [`evals/`](../../evals/) (`summary_probe.py`, `memory_probe.py`), [`backend/tests/`](../../backend/tests/)

---

### 1. Tests check the plumbing; evals check the model
- **Concept:** **Tests** use fake models, run in milliseconds, are deterministic, and check that *your code* works. **Evals** use the real model and measure how *good* its behaviour is.
- **In this repo:** `backend/tests/` uses `FakeListChatModel`, `RunnableLambda` writers, fake models scripted with `tool_calls`, and `KeywordEmbeddings`. CI runs them on Ubuntu and Windows. `evals/` needs Ollama, so it runs locally.
- **Why it matters:** "All tests pass" says nothing about whether the summaries are any good. You need both.

### 2. A prompt change is a guess until you measure it
- **Concept:** You can't tell from one or two examples whether a prompt change helped.
- **In this repo:** "Reply in the user's language" (v1.1) didn't stop the CJK leak. "Don't call tools for small talk" changed nothing (75% → 75%). "Don't overclaim a trend" scored 0/3. All three were only discovered by measuring.
- **Why it matters:** Most of the project's real fixes ended up in code. You only find out which rules to move into code by measuring the prompt version first.

### 3. The loop: run everything, change one thing, run everything again
- **Concept:** Measure → change one thing → re-measure the **whole** suite → keep the change only if the numbers say so. Record what produced each result.
- **In this repo:** `uv run python evals/run_evals.py [--only routing,retrieval,extraction] [--runs 2]` prints a scorecard with the change vs the previous run, and saves `evals/results/evals-<time>.json` with the model, `PROMPT_VERSION` and git commit (`+dirty` if there are uncommitted changes).
- **Why it matters:** Prompt v1.4 fixed its target case and **broke another one**. Running only the failing case would have hidden that.

### 4. Label the correct answer, and review the labels
- **Concept:** Datasets hold the *correct* answer, not what the model does today, so known failures show up as failures. Labels can also be wrong.
- **In this repo:** `evals/datasets/*.jsonl` (routing 45, retrieval 60 entries + 30 questions, extraction 25). Mixed authoring: hand-written for tricky and known-failure cases, AI-drafted for the bulk, then reviewed by a human.
- **Why it matters:** One label expected `mood: null`, but our own schema says null only applies when there are no events, so the *eval* was wrong. Mood labels became *ranges* (e.g. 5–7), because demanding exactly "6" measures noise. Change a label only with a written reason; otherwise you're moving the goalposts.

### 5. Check the checker
- **Concept:** The scoring code has bugs too. Unit-test it, and sanity-check the datasets.
- **In this repo:** `evaluators.py` is pure functions (`check_extraction`, `matches`, `confusion`, `recall_at`, `mrr`) tested in `evals/tests/test_evaluators.py`. `test_datasets.py` catches label typos (an unknown route, a gold date that isn't in the journal). Both run in CI.
- **Why it matters:** In M4 and M5, several "failures" were the keyword checker missing a correct answer ("that dinner with your sister on Wednesday"). A wrong checker can send you off fixing a model that was right.

### 6. Score only what a case is about, and split your metrics
- **Concept:** A case should fail only for the reason it was written for. One "accuracy" number hides opposite failure modes.
- **In this repo:** An extraction case defines only the checks it tests (`mood`, `placement`, `forbidden`, …). Tool use is split into **missed-call** vs **false-call** rates; routing reports a confusion matrix; retrieval reports recall@k and MRR separately from the "nothing found" cutoff.
- **Why it matters:** llama3.1 *over*-calls tools; qwen3 *under*-calls them. A single accuracy figure would have made them look similar.

### 7. Isolate components, cheapest first, and look inside the pipeline
- **Concept:** Evaluate each stage on its own, so a failure points at one component. Run the cheap suites first.
- **In this repo:** Routing (no LLM, instant) → retrieval (embeddings only, seconds) → extraction (chat model, ~1 min) → tools (one real graph turn per case) → language (free: it scans every text the other suites generated for CJK leaks). The slow `judge` and `simulation` suites are opt-in (`--only judge,simulation`; see [note 9](09-llm-judges-and-simulation.md)). The summary probe saves intermediate digests and counts calls per stage.
- **Why it matters:** In M4, a model-free retrieval check proved search was fine (4/4), so the remaining failures were reasoning. The saved digests showed a summary error came from the *map* step, not the reduce.

### 8. Noise: repeat runs, and compare variants side by side
- **Concept:** Outputs vary even at temperature 0, and they can shift *between sessions*. Compare variants back to back, with repeats.
- **In this repo:** In M7, a "today only" prompt rule seemed to make all-checks drop from 74% to 70%. But with the prompt reverted, the same events still dropped: the baseline from an earlier session wasn't comparable. A fair A/B (each variant twice × 3 runs, back to back) gave 72% (code fix only) vs 71% (code fix + prompt rule). The code fix was kept.
- **Why it matters:** Comparing against an old baseline almost recorded the wrong cause.

### 9. Real usage writes your best test cases, and 100% means the eval is too easy
- **Concept:** Failures you see in real use become eval cases. A perfect score means you need harder cases, not that the model is perfect.
- **In this repo:** `assistant_embellishes` came from a real chat where the assistant's word "invigorating" became the user's mood. The M2 probe hit 100% while still missing "8 out of 10" listed as a highlight and a `work` tag on a dentist trip. The M7 suite now catches both.
- **Why it matters:** With 10–25 cases, a single label moves a rate by 2–4 points. Treat numbers as direction, and keep growing the dataset. **The dataset is the product.**

---

### Try it
- Run `uv run python evals/run_evals.py --only extraction --runs 3` twice in a row, then once after `ollama stop llama3.1:8b`. Do the numbers move between sessions?
- `uv run python evals/run_evals.py --only routing`, then add an exception for "check in at" to `CHECKIN_PHRASE` and rerun.
- Add your own tricky transcript to `evals/datasets/extraction.jsonl` and run `--only extraction`.

**Full story:** [NOTES.md](../NOTES.md) → M1 (first eval), M2 ("The eval loop in practice"), M7 (harness, baseline, "First fix through the loop"). Next: [note 9](09-llm-judges-and-simulation.md) covers the LLM judge and simulated users (M7 steps 6–7).
