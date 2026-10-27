# 9. LLM judges and simulated users

[Note 8](08-evals.md) covered evals that code can score. This note covers the two harder tools: an **LLM judge** for qualities only a reader can assess (tone, follow-up questions, groundedness, safety), and a **simulated user** that runs whole conversations through the real agent. Both are powerful, and both are easy to over-trust. Most of this note is about how not to.

**Where to look:** [`evals/judge.py`](../../evals/judge.py) (rubrics, `Verdict`, `JUDGE_MODEL`, `CALIBRATED`), [`evals/judge_calibrate.py`](../../evals/judge_calibrate.py), [`evals/suites/judge.py`](../../evals/suites/judge.py), [`evals/simulator.py`](../../evals/simulator.py), [`evals/suites/simulation.py`](../../evals/suites/simulation.py), [`evals/evaluators.py`](../../evals/evaluators.py) (`count_questions`, `agreement`, `in_character`, `said_gold`), [`evals/datasets/`](../../evals/datasets/) (`chat.jsonl`, `judge_planted.jsonl`, `judge_gold.csv`, `personas.jsonl`)

---

### 1. Use a judge only for what code can't check
- **Concept:** An LLM judge is slower, noisier and less trustworthy than code. Use it only where no deterministic check exists.
- **In this repo:** `judge.CHECKS = ("kind", "follow_up", "grounded", "safety")`. "Ask only ONE question" *is* checkable, so it's `one_question` in code (`evaluators.count_questions`), and the judge is told not to count.
- **Why it matters:** The same principle as guardrails ([note 2](02-structured-output.md), point 9), applied to grading. Every check you move into code is one the judge can't get wrong.

### 2. Rubrics for a small judge: one narrow pass/fail question, reasoning first
- **Concept:** An 8B judge grades one specific yes/no question far more reliably than a 1–10 score over several qualities at once.
- **In this repo:** One rubric per check in `judge.RUBRICS`, each with explicit PASS and FAIL examples ("at least…" is minimising; "you should be proud" is fine). `Verdict` has `reasoning` *before* `passed`, so the model explains before it decides (field order = generation order, as in [note 2](02-structured-output.md), point 6). Temperature 0, and bad JSON is retried with `invoke_structured`.
- **Why it matters:** Vague rubrics measure the judge's own taste. Concrete examples pin down what you actually mean.

### 3. Pin the judge model and version the rubric
- **Concept:** The judge is a ruler. If it changes whenever the agent changes, old and new scores aren't comparable.
- **In this repo:** `JUDGE_MODEL = "llama3.1:8b"` is pinned in `judge.py`, separate from the agent's `DIARY_CHAT_MODEL`. `RUBRIC_VERSION` (now `r3`) is bumped whenever a rubric changes, like `PROMPT_VERSION`. The simulator is pinned the same way (`SIM_MODEL`).
- **Known risk:** Here the judge is the same model as the agent. That risks *self-preference* (rating its own style highly) and shared blind spots. Calibration (next point) is how you find out how bad that is.

### 4. Calibrate against hand grades before trusting a single score
- **Concept:** A judge's score is only worth what its agreement with a human says. Measure that first.
- **In this repo:** `judge_calibrate.py sample` freezes real replies plus hand-written bad ones into `judge_gold.csv`, shuffled, and a human grades them blind (y/n). `judge_calibrate.py agree` runs the judge 3× per cell (majority vote) and reports agreement. A check is added to `judge.CALIBRATED`, and counts on the scorecard, only once agreement is ≥ 85% and fail_catch is ≥ 80%. Until then, the scorecard shows it as **unverified**.
- **Why it matters:** The first round had `grounded` at **38%** agreement and `safety` at **50%**. An uncalibrated judge would have produced confident-looking numbers that meant nothing.

### 5. Plain agreement hides a lazy judge, so measure fail_catch and plant failures
- **Concept:** If 90% of replies are fine, a judge that *always* says "pass" scores 90% agreement and catches nothing. Measure the failure side separately.
- **In this repo:** `evaluators.agreement` reports **fail_catch** (of the replies the human failed, how many did the judge fail?) and **false_alarm** (of the human's passes, how many did the judge fail?). `judge_planted.jsonl` holds 9 hand-written bad replies, because the agent's own replies rarely fail, and you can't measure catching failures you never show the judge.
- **Why it matters:** This is the same idea as splitting missed calls from false calls ([note 8](08-evals.md), point 6), applied to the judge itself.

### 6. Most disagreements are about definitions, and calibration changes the human too
- **Concept:** When the judge and the human disagree, it's often because they meant different things, not because the judge is random.
- **In this repo:** The judge gave the same verdict on 95–100% of repeats; the disagreements came from the definitions. In round 1 → r2:
  - **safety:** the *judge* was right. The agent's replies to "I've been thinking about hurting myself again" were warm but never suggested reaching out. Two hand grades changed, and a real product gap was found.
  - **grounded:** the rubric widened to match the human: invented *feelings* ("last week was overwhelming") and wrong times now fail.
  - **follow_up** gained a rule: after hard news, the question must stay with it.
- **Why it matters:** Calibration pins down what *you* mean as much as it tests the judge. r2 agreement: follow_up 95%, grounded 100%, safety 100%, kind 75%.

### 7. A calibration only covers the kind of data it was measured on
- **Concept:** A judge tuned on one set of replies can fail on a different kind of reply. And re-scoring the rows you tuned on overstates agreement (overfitting).
- **In this repo:** On simulated check-ins, `grounded` r2 failed ordinary reflection ("it sounds like the rent is weighing on you"), passing only 19% of turns. Round 1 had never graded a check-in reflection. r3 allowed restating a feeling the user expressed. A second round with 12 new check-in rows then dropped grounded agreement from **100% → 65%**, and follow_up from 95% → 84%. The 8B judge sometimes misreads the conversation, e.g. it said a marathon wasn't mentioned when the user had mentioned it one line earlier.
- **Why it matters:** `CALIBRATED` is still empty, and the judge scores stay labelled unverified. The next step is a fresh blind batch, not more tuning on the same rows. `kind` stays unverified for the same reason: more rubric tweaks on the same 28 rows would be overfitting.

### 8. Simulated users run whole flows through the real agent
- **Concept:** A second LLM plays a user with a persona and a scripted "true day", talks to the real agent, and you check the final result against the script.
- **In this repo:** `simulator.py → sim_reply()` plays one of 6 personas from `personas.jsonl` (talkative, terse, rent stress, grief, mentions a past day, asks a lookup mid check-in) through a whole `/checkin` (`suites/simulation.py → converse`). The code approves the save, and the entry in journal.db is checked with the same `check_extraction` used by the extraction suite. Each check-in question is also judged for follow_up and grounded. `inject` pins the line a test is about, so the key moment doesn't depend on the simulator improvising it.
- **Why it matters:** Component evals test each piece alone; this tests what users actually experience, end to end.

### 9. Guard against the simulator's own failures
- **Concept:** The simulated user is an LLM too. It forgets its script, slips out of character and improvises, and those errors must not be counted as the agent's.
- **In this repo:**
  - The simulator forgot the script, and a run scored "entry mentions sushi" when sushi was never said. `evaluators.said_gold` now scores only what the simulator actually said, and `avg_gold_unsaid` reports how much it skipped.
  - Runs where it slips into the assistant's voice ("how can I help?") are dropped by `evaluators.in_character`, not scored.
- **Why it matters:** The same model on both sides is more cooperative and articulate than real users, so expect real conversations to go worse. It also improvises wrong details (it said "Monday" on a Thursday). Read transcripts, not just scores.

### 10. Whole-flow tests find rules that clash
- **Concept:** Rules that are each correct can conflict once they meet in a real flow. Only an end-to-end test sees that.
- **In this repo:** M2's gratitude guardrail ("gratitude must be explicit": the user must say grateful/thankful) met M3's check-in, which *asks* "what are you grateful for?". The user's answer, "my sister offered to help", was deleted in **5 of 12** simulated entries. Fix: `extraction.asked_for_gratitude()` counts an answer to a gratitude question as explicit. Entry all-checks went 25% → 50% (one run each, so read it as direction).
- **Why it matters:** Unit tests and component evals all passed. Simulation also surfaced product issues logged as #13 (check-in drifts off its topics), #14 (lookups mid check-in get invented answers) and #15 (jumps to the next topic right after hard news).

---

### Two dependencies we chose not to add
- **`agentevals`** compares the sequence of tool calls. The checks that found real bugs here were about the *arguments* ("does the date range cover last Monday?"), so custom checks (`evaluators.check_tool_turn`, `covers`) fit better.
- **`openevals`**' LLM judge is a prompt plus structured output, without the retry-on-bad-JSON that an 8B model needs, which `invoke_structured` already does.

A dependency should earn its place.

### Try it
- `uv run python evals/judge_calibrate.py sample`, grade a few rows in `judge_gold.csv` yourself, then `agree`. Where do you and the judge disagree, and is it the judge or the definition?
- Delete the planted rows from the calibration and watch fail_catch become meaningless.
- `uv run python evals/run_evals.py --only simulation` and read one full transcript in the results JSON, not just the scores.
- Remove `asked_for_gratitude` from `apply_guardrails` and rerun `--only simulation`: count the entries with empty gratitude.

**Full story:** [NOTES.md](../NOTES.md) → M7 step 6 ("an LLM judge, calibrated before it's trusted", "Calibration round 2") and M7 step 7 ("simulated users run whole check-ins").
