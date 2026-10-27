# Repo learnings (AI-maintained)

last_synced_commit: 671898d
last_synced: 2026-10-27
current_milestone: M7 steps 0–7 built; judge round 2 done but `CALIBRATED` still empty (r4 + fresh batch next); M8 optional

> Maintained by the `learn-repo` skill. The code map and concept index live here. The narrative, experiments, and eval tables live in `docs/NOTES.md`. Roadmap lives in `docs/PLAN.md`. Topic-organised notes for other learners live in `docs/learning/` (01–10 + README, by this skill; 09 = judge/simulation, 10 = principles capstone). Keep them in step when a lesson changes (e.g. a fix supersedes one in 09's table).

## End-to-end flow (as of M7 step 3)

Layers: `cli/` → `services/` → `repositories/` → `models/`; `core/` (config, llm, prompts) is used everywhere. Paths below are under `backend/src/diary_agent/`. The full ASCII graph is drawn in the `graph.py` module docstring.

```
uv run diary chat  (cli/app.py → chat())
  ├─ DiaryContext(chat_model, extractor, conn, vectors, day, model_name, memory_extractor)   deps, never checkpointed
  ├─ build_graph(open_checkpointer(), store=open_store())   SqliteSaver → checkpoints.db; SqliteStore → store.db
  ├─ thread_config(day)                 thread_id = today's date → resume after restart
  └─ each user line → run_turn → stream_turn(graph, {"user_input": text}) = ONE graph run
       START → router (rule-based: /save /summary /cancel /checkin, "check in" keyword, mode)
         ├─ chat_agent ⇄ tools   (ReAct loop, M4)
         │    system = COMPANION_SYSTEM + memory_block(store, user_input) (M6) [+ TOOLS_SYSTEM(calendar_hint)]
         │    tools_for(text): MEMORY_REQUEST → TOOLS | LOOKUP_REQUEST → READ_TOOLS | else none → plain invoke → END
         │    tools_condition: tool_calls? → ToolNode → route_after_tools:
         │         pending_memories? → confirm_memory (interrupt per proposal, write, swap placeholder
         │         ToolMessage by id) → chat_agent;  else → chat_agent
         │    no tool_calls (or MAX_TOOL_ROUNDS=3 used, tool_calls dropped) → END
         │    tools: search_entries (Chroma, cutoff 0.43 → dates → SQLite) | get_entries_by_date (SQLite)
         │           | remember_fact (validates content, propose() → Command(pending_memories + ToolMessage))
         ├─ checkin:    + CHECKIN_SYSTEM(topic = CHECKIN_TOPICS[step]), step+1 → END   (no tools)
         ├─ summarize:  notice "run 'diary summary'" → END   (kept out of chat: would pollute today's transcript)
         ├─ done (/cancel) → END
         └─ extract: format_transcript (drops tool msgs + assistant replies after a lookup) → extract_entry → draft dict
              └─ confirm: interrupt({draft, exists}) ── pause + checkpoint
                   CLI: pending_interrupt → answer_interrupt (ask_to_save | ask_to_remember) → run_turn(Command(resume=...))
                   ├─ True  → save: SQLite save_entry, then vector_repo.upsert_entry (failure → "run reindex")
                   │          → update_memory: extract_memories (LLM + apply_memory_guardrails) → propose (dedupe)
                   │             → proposals? confirm_memory (resume = list[bool]) → END ; none/failure → END
                   └─ False → END ("Not saved.")
       stream_mode=["messages","updates"]: tokens only from REPLY_NODES; updates → status
         ("Writing up your entry...", "Searching your journal...")
  exit → unsaved (len(messages) > saved_count)? → offer /save
uv run diary list / show / reindex (rebuild Chroma from SQLite) / memory list|add|forget
uv run python evals/run_evals.py [--only routing,retrieval,extraction] [--runs N]   (M7 scorecard)
uv run diary summary --week|--month [--date D] [--saved]   (cli/app.py → summary(), no graph)
  period_range (calendar Mon–Sun / month) → list_entries(start, end) → none? stop, no LLM call
  summarize_period(writer, reducer, entries, ...)   all calls num_ctx=8192, temperature 0
    map:      day_line(e) per entry, in code (no LLM): "Mon 2026-10-19 (mood 4, felt tired): <summary>
              Highlights: ... Challenges: ..." from the approved fields; transcript never read
    collapse: only if digests > REDUCE_INPUT_TOKENS; _groups packs consecutive digests; ≤ MAX_COLLAPSE_ROUNDS
    reduce:   REDUCE_SYSTEM + field_guide(PeriodSummary) + stats_text (code: avg, low/high day, tags 2+ days)
              → invoke_structured (retry w/ feedback) → PeriodSummary
    apply_guardrails: drop "mood" wins/challenges; < MIN_TREND_DAYS rated → code writes mood_trend
  → SummaryRecord → render_summary → Confirm → summary_repo.save_summary (journal.db `summaries`)
```

## File map

| File | What it does | Key symbols | Concepts it teaches |
|---|---|---|---|
| `core/config.py` | Env/.env-driven frozen settings | `Settings`, `settings`, `PROJECT_ROOT` | Config in one place; model choice is config, not code |
| `core/llm.py` | Model factories | `get_chat_model(temperature, max_tokens, num_ctx)`, `get_embeddings()` (nomic-embed-text, 768-dim) | Depend on `BaseChatModel`; provider swap = one file; `num_ctx` None = Ollama default, overflow cut silently |
| `core/prompts.py` | System prompts + version | `PROMPT_VERSION` (v1.10), `COMPANION_SYSTEM`, `CHECKIN_SYSTEM`, `TOOLS_SYSTEM`, `EXTRACT_SYSTEM`, `COLLAPSE_SYSTEM`, `REDUCE_SYSTEM`, `MEMORY_SYSTEM`, `EXTRACT_MEMORY_SYSTEM` | Prompt versioning; TOOLS_SYSTEM appended only when tools are bound; `{calendar}`/`{memories}` filled by code |
| `services/agent/state.py` | Graph state + run deps | `DiaryState(MessagesState)` (+ `pending_memories` with `merge_pending` reducer), `DiaryContext` (+ `vectors`, `memory_extractor`), `Route` | State vs context; `add_messages` appends *and replaces by id*; custom appending reducer for parallel tool writes |
| `services/agent/nodes.py` | Node functions | `router`, `chat_agent`, `tools_for`, `LOOKUP_REQUEST`, `MEMORY_REQUEST`, `calendar_hint`, `_companion_prompt` (adds `memory_block`), `MAX_TOOL_ROUNDS`, `GAVE_UP`, `checkin`, `extract`, `confirm`, `save`, `update_memory`, `confirm_memory`, `route_after_tools` | Code picks the tool *menu* per turn; code does date math; loop budget; memory never blocks save; write after `interrupt()` is OK |
| `services/agent/tools.py` | Tools; none writes | `search_entries`, `get_entries_by_date`, `remember_fact` (returns `Command`), `format_entry`, `MAX_DISTANCE`, `SEARCH_K`, `READ_TOOLS`, `TOOLS` | `@tool` docstring + `Annotated` args = the whole schema; `ToolRuntime` (context, state, store, tool_call_id) hidden; errors as strings; a tool stages state via `Command` + its own ToolMessage |
| `services/agent/graph.py` | Wiring + runtime helpers | `build_graph(checkpointer, store)`, `stream_turn`, `pending_interrupt`, `REPLY_NODES` | Loop = conditional edge + edge back; `compile(store=)` → `runtime.store`; status "Looking for things to remember..." |
| `models/memory.py` | Memory schemas | `MemoryItem` (evidence, kind, content), `MemoryCandidates` (max 5), `Memory` (+ id, source_date, updated_at), `MemoryKind` | Same two-layer + evidence-first pattern as entry.py |
| `repositories/memory_repository.py` | LangGraph `SqliteStore` (data/store.db) | `open_store` (index on `content`, 768 dims), `NAMESPACE=("user","profile")`, `put_memory` (falls back to `index=False`), `list_memories`, `search` → (memory, similarity), `delete_memory`, `new_id` | Long-term memory across threads; similarity (higher = closer) vs Chroma distance |
| `services/memory.py` | Memory write policy + read injection | `apply_memory_guardrails` (evidence ≥0.8 user words, `TRANSIENT`, 120 chars, `THIRD_PERSON` rule), `extract_memories`, `propose` (`DUPLICATE_SCORE` 0.90 skip, `RELATED_SCORE` 0.70 same kind → update), `apply_proposals`, `select_for_prompt` (`MAX_INJECTED` 15, else goals + top-5 ≥0.5), `memory_block` | Memory pollution; precision before recall; thresholds chosen by error cost |
| `repositories/vector_repository.py` | Chroma index, one vector per day | `open_store` (cosine, telemetry off), `entry_text`, `upsert_entry` (id = date), `search` → `(date, distance)`, `_day_key`, `DOC_PREFIX`/`QUERY_PREFIX`, `count` | Vector store as a rebuildable index, SQLite is truth; embed extracted fields, not transcript; nomic task prefixes; numeric date metadata for `$gte` |
| `models/entry.py` | Pydantic schemas | `ExtractedEntry` (LLM-inferred), `JournalEntry` (+ code-known fields) | Two-layer schema; all fields required; bounded lists; evidence before judgement |
| `services/extraction.py` | Transcript → entry | `build_extractor`, `extract_entry`, `invoke_structured` (shared retry loop), `apply_guardrails`, `asked_for_gratitude` (answer to a gratitude question = explicit), `field_guide(schema)`, `format_transcript` | Structured output, retry-with-feedback, guardrails; transcript drops tool msgs and the assistant reply after a lookup (M7 past-day leak) |
| `models/summary.py` | Summary schemas | `PeriodSummary` (LLM-written: themes, wins, challenges, mood_trend, overview), `SummaryRecord` (+ range, count, `mood_avg`, model, version), `Period` | Same two-layer pattern as entry.py; numbers are code-known |
| `services/summary.py` | Entries → review, map-reduce | `summarize_period`, `day_line`, `collapse`, `_groups`, `reduce`, `apply_guardrails`, `stats_text`, `period_range`, `mood_avg`, `estimate_tokens`, `build_writer`/`build_reducer`/`get_summary_model`, `NUM_CTX`, `REDUCE_INPUT_TOKENS`, `MAX_COLLAPSE_ROUNDS`, `MIN_TREND_DAYS` | Map in code from approved fields; token budget (chars/4); collapse as last resort; code computes stats; guardrails override sparse trend |
| `repositories/entry_repository.py` | SQLite repo | `connect` (`check_same_thread=False`), `save_entry`, `get_entry`, `list_entries(start, end)` | Lists as JSON text; pydantic round-trip |
| `repositories/summary_repository.py` | SQLite summaries in journal.db | `save_summary`, `get_summary(period, start)`, `ensure_schema`; PK `(period, start_date)` | Reuses `entry_repository.connect`; table created on first use |
| `cli/app.py` | Typer CLI, thin graph driver | `chat`, `run_turn`, `answer_interrupt` → `ask_to_save` / `ask_to_remember` (list[bool]), `list_cmd`, `show`, `reindex`, `summary`, `render_summary`, `memory_app` (`list`, `add`, `forget`) | CLI answers interrupts; resume-on-start; `reindex` repairs the index; every write needs a yes |
| `evals/run_evals.py` | M7 harness: one scorecard, deltas vs the previous run, results JSON (+ commit, `+dirty`) | `SUITES`, `scorecard`, `previous_metrics`, `kind` (n_* counts, avg_*/*mrr/*distance numbers, rest rates), `report`; language runs on `texts` the other suites return | Measure → change one thing → re-measure |
| `evals/evaluators.py` | Pure scoring functions | `check_extraction` (only scores checks a case defines), `matches` ("a\|b"), `confusion`, `rank_of`, `recall_at`, `mrr`, `check_tool_turn`, `covers`, `has_cjk`/`CJK`, `ADMITS`, `count_questions`, `agreement` (agree / fail_catch / false_alarm), `in_character`, `said_gold` | Test the checker (evals/tests/, runs in CI) |
| `evals/suites/` + `datasets/*.jsonl` | routing (45, no LLM), retrieval (60 entries + 30 q, embeddings), extraction (26, replaced extract_probe), tools (14, real graph, today = 2026-10-22, replaced tool_probe), language (free, scans generated texts); opt-in: judge (`chat.jsonl` 19 + `judge_planted.jsonl` 9), simulation (`personas.jsonl` 6) | `run(runs) -> {metrics, cases, failures, texts?}`, `load_jsonl`, `seed_journal` | Labels = correct intent, so known misses fail; `test_datasets.py` catches label typos |
| `evals/summary_probe.py` | M5 summary probe, 3 cases × N via real `summarize_period` | seeds `datasets/tools_journal.jsonl` via `seed_journal`; `spy()` counts map/collapse/reduce by system prompt; checks grounded/mood/recurring/embellish/sparse/clean; `--collapse` | Inspect intermediate outputs (digests), not just the final answer |
| `evals/judge.py` | LLM judge | `JUDGE_MODEL` (pinned llama3.1:8b), `RUBRIC_VERSION` (r3), `CHECKS` (kind, follow_up, grounded, safety), `RUBRICS`, `Verdict` (reasoning before `passed`), `view`, `grade`, `CALIBRATED` (empty) | One narrow pass/fail per call; pinned ruler; unverified until calibrated |
| `evals/judge_calibrate.py` | Hand grades vs judge | `sample` → `datasets/judge_gold.csv` (blind, shuffled), `add_sim` (check-in rows), `agree` (3 runs, majority) | Calibration before trust; fail_catch beats raw agreement |
| `evals/simulator.py` + `suites/simulation.py` | Simulated user through a full `/checkin` | `SIM_MODEL` (pinned), `SIM_SYSTEM`, `sim_reply`, `converse`, `inject`, `MAX_ANSWERS`, `JUDGED` | Whole-flow test; code approves save; entry scored with `check_extraction` on what the sim actually said |
| `evals/memory_probe.py` | M6 memory probe: `--part extract\|similarity\|tool` | 8 transcripts, `PAIRS` similarity calibration, tool choice | Candidates vs post-guardrail; calibrate thresholds on measured scores |
| `backend/tests/services/test_memory.py`, `agent/test_memory_graph.py`, `repositories/test_memory_repository.py` | Memory tests on `InMemoryStore` | — | Guardrails, propose/dedupe, interrupt + placeholder swap through the graph |
| `backend/tests/services/test_summary.py`, `repositories/test_summary_repository.py` | Summary plumbing tests | — | RunnableLambda writer + scripted reducer; budget logic, ordering, guardrails, no-entries = no call |
| `backend/tests/conftest.py` | Shared fakes | `KeywordEmbeddings`, `embeddings` fixture | Bag-of-words fake so search *ranking* is testable without Ollama |
| `backend/tests/services/agent/test_tools.py`, `test_chat_helpers.py`, `repositories/test_vector_repository.py` | Tool, gate/calendar, Chroma tests | — | Tools tested directly with a fake runtime; cutoff and upsert-replaces behaviour |
| `backend/tests/services/agent/test_graph.py` | Graph behaviour | restart, interrupt, check-in, tool loop, tool msgs kept out of extraction | Fake models scripted with tool_calls; test via `stream_turn` |
| `.github/workflows/ci.yml`, `dependabot.yml` | CI: pytest on ubuntu + windows; privacy guard | `privacy` job greps `git ls-files` for data/, *.db, chroma/, .env | Evals need Ollama, so CI runs only fake-backed tests; a diary must never be published |
| `scripts/smoke.py` | Ollama connectivity check | — | `.stream()`, embedding dim |

## Concept index

### M0–M1: Models, LCEL, history (compressed)
- Chat model takes/returns messages; invoke vs stream; embeddings = text → vector (`core/llm.py`).
- LCEL `prompt | model | parser`; every Runnable has invoke/stream/batch.
- Model is stateless: "memory" = resending history (now `DiaryState.messages`; trimming in M6).
- Tests (fakes, plumbing) vs evals (real model quality). First eval = CJK leak → llama3.1. NOTES M1.

### M2: Structured output and guardrails
- **`with_structured_output(method="json_schema", include_raw=True)`** → `{raw, parsed, parsing_error}` enables retry with feedback. `build_extractor`.
- **Grammar-constrained decoding** guarantees structure, not content; only *required* fields forced; descriptions not seen → `field_guide`.
- **Deterministic guardrails > prompt tweaks** for code-checkable rules. `apply_guardrails`.
- **Never ask the model for what code knows** (date, transcript, model, version). `JournalEntry`.
- **Eval hygiene:** rerun everything after a change; labels can be wrong; temp 0 ≠ deterministic. NOTES M2.

### M3: LangGraph (state, nodes, edges, checkpoints, interrupts)
- **StateGraph** = nodes `(state, runtime) -> partial update` + edges. `build_graph`.
- **State vs context:** serializable data (checkpointed) vs deps (`runtime.context`). Rule: survives a restart → state.
- **Reducers:** `messages` appends; other keys last-write-wins.
- **Conditional edge vs `Command(goto=, update=)`.** Router + edge for visible routing; `confirm` decides + updates.
- **Multi-turn without a loop:** `mode`/`checkin_step` in state; code picks topic, model words it.
- **Checkpointer = short-term memory:** `SqliteSaver`, thread_id = date.
- **`interrupt()` = HITL;** resume with `Command(resume=…)`; **node reruns from the top**, so writes live in `save`.
- **Streaming filter:** `meta["langgraph_node"] in REPLY_NODES`. Worker-thread SQLite bug only showed in `stream`. NOTES M3.

### M4: Tools, ReAct, RAG, tool evals
- **ReAct loop = graph cycle:** `chat_agent` → `tools_condition` → `ToolNode` → back. Still one invoke per user turn. `build_graph`.
- **A tool is name + docstring + arg schema; that's all the model sees.** Docstring says *when* to use it. `ToolRuntime` injects `DiaryContext`, hidden from schema. `tools.py`.
- **Narrow args, validated in code;** errors returned as strings so the model can retry (`ToolNode` also catches exceptions). Optional date args on search got junk (`0000-00-00`) → removed.
- **Read-only tools on purpose:** no `save_entry` tool; writes stay behind `interrupt()`.
- **RAG:** embed on save (extracted fields, not transcript), id = date (upsert), search returns dates, SQLite returns entries. Index is rebuildable (`diary reindex`); index failure doesn't fail the save.
- **kNN always returns k → need a relevance cutoff** (`MAX_DISTANCE`), or the model "finds" a ski trip in a pottery entry.
- **Move decisions the model is bad at into code:** `calendar_hint` (date math: 50% → 90% args) and `LOOKUP_REQUEST` gate (false tool calls 75% → 0%; the prompt rule changed nothing).
- **Loop budget:** `MAX_TOOL_ROUNDS = 3`; then call without tools and strip any tool_calls.
- **Keep tool traffic out of the entry:** `format_transcript` drops tool messages, or past days leak into today.
- **Measure tool use with split rates:** missed-call vs false-call (llama over-calls, qwen3 under-calls). Model-free retrieval check (recall 4/4) separates search from reasoning. NOTES M4.

### M5: Map-reduce summaries, context budgeting
- **Map-reduce:** map = `day_line` per entry (was an LLM condense via `writer.batch()`, LCEL's map; replaced 2026-10-26 because extraction already condensed the day and the user approved it); collapse only when over budget; reduce = structured `PeriodSummary`. `summarize_period`.
- **Context window is a budget:** Ollama's default window cuts overflow *silently*. One fixed `NUM_CTX = 8192` for all summary calls (changing it reloads the model). Tokens estimated as chars/4.
- **Don't redo a step you already have:** the LLM map re-condensed transcripts that extraction had already condensed. Dropping it: same checks, sparse 2/3 → 3/3, month 15 s → 4.3 s. A stored verbatim `digest` was tried and dropped (fragments lost context).
- **Code computes the numbers** (avg, low/high day, recurring tags) → `stats_text`; code writes the digest date/mood prefix.
- **But more computed facts aren't free:** a "Mood trend: down" line made the review skip low/high days (2/6 vs 6/6) → removed.
- **Length demands cause invention:** "2-3 sentences" padded a short entry ("first project is now complete") → "at most 3, one is fine" (that prompt was later removed with the LLM map step).
- **Prompt rule failed again (0/3)** on sparse "steady throughout" → `apply_guardrails` writes `mood_trend` when < `MIN_TREND_DAYS`; also drops "good mood" as a win.
- **Collapse has a floor:** forced 150-token budget → 9 extra calls, 2.3× slower, still over budget after 3 rounds; dates lost. `MAX_COLLAPSE_ROUNDS`.
- **Errors compound across hops:** the pottery distortion came from *map*; reduce repeated it. Probe saves digests. NOTES M5.

### M6: Long-term memory
- **Checkpointer (one thread/day) vs store (all threads):** `SqliteStore` with a semantic index, `compile(store=)` → `runtime.store`; not in `DiaryContext`.
- **Read path = system prompt, never `messages`** (would leak into transcript → entry). `memory_block`, `select_for_prompt`.
- **Two write paths, one gate:** `update_memory` after save, `remember_fact` tool; both only *propose*; `confirm_memory` interrupts per proposal.
- **Tool stages, doesn't write:** `remember_fact` returns `Command(update={messages: [placeholder ToolMessage], pending_memories})`; `confirm_memory` swaps the ToolMessage *by id*. No `interrupt()` in tools (ToolNode thread pool).
- **Write after `interrupt()` in the same node is fine** (only the resumed run gets past it) vs `confirm → save` split. Rule: nothing *before* `interrupt()` has side effects.
- **Memory pollution:** a bad memory hits every future prompt → confirm everything, guardrails lean to dropping. Wrong-subject bug ("Lives in Leeds" from sister's move) → `THIRD_PERSON` rule.
- **Prompt too cautious then too eager:** recall 6/12 → 5/6; eagerness fixed in code.
- **Per-tool gate:** `MEMORY_REQUEST` (imperative only) → false saves 4/4 → 0.
- **Similarity = topic, not identity:** updates 0.55–0.75 overlap different facts 0.38–0.58 → thresholds avoid overwriting (lose data); missed update = 2 memories. NOTES M6.

### M7 (steps 0–3): Eval harness
- **One command, one scorecard, deltas vs last run;** results JSON tagged model + prompt + commit (`+dirty`). `run_evals.py`.
- **Cheapest first:** routing (no LLM) → retrieval (embeddings) → extraction (LLM).
- **Labels = correct answer;** mixed hand/AI-drafted, human-reviewed. **Test the checker** (`evals/tests/`, in CI). Score only the checks a case defines.
- **Baseline:** routing 82% / tool gate 62%; retrieval recall@1 96%, MRR 0.97 but cutoff margin 0 (0.41 vs 0.41); extraction all-checks 74% (`not_in`, `forbidden` 50%).
- **Between-session drift at temp 0:** compare variants back to back with repeats. A/B 72% (code) vs 71% (code + prompt) → kept the code fix in `format_transcript`.
- **Open:** issue #9 past-day recall without a lookup still extracted as today. NOTES M7.

### M7 (steps 4–7): tools, language, judge, simulation
- **Tools suite** = tool_probe folded in (checks → `check_tool_turn`, unit-tested): choice 98%, args 87%, false calls 0%. Checker fix: "8 days since" was correct. No `agentevals` (bugs were in *args*, not call order).
- **Language suite is free:** scans `texts` other suites returned; 120/120 clean.
- **Judge only for what code can't check;** `one_question` is code. One rubric per check, pass/fail, reasoning first, temp 0. No `openevals` (no bad-JSON retry).
- **Calibrate before trusting:** blind hand grades, judge 3× majority; agreement + **fail_catch** + false_alarm; planted bad replies because real ones rarely fail. Gate: agree ≥ 85% and fail_catch ≥ 80% → `CALIBRATED`.
- **Disagreements were definitions** (judge 95–100% consistent): safety judge was right (no reach-out = product gap); grounded widened (invented feelings). r1 → r2 grounded 38% → 100%.
- **Calibration has a range / overfitting:** grounded r2 failed check-in reflection (19%) → r3; round 2 on new check-in rows: grounded 100% → 65%, follow_up 95% → 84%; judge misreads the conversation (8B limit). `kind` unverified.
- **Simulator pitfalls:** same model = too cooperative; forgets script → `said_gold`/`avg_gold_unsaid`; slips into assistant voice → `in_character` drops the run; `inject` pins the key line.
- **Whole-flow test found a rule clash:** M2 gratitude guardrail × M3 check-in question deleted real gratitude 5/12 → `asked_for_gratitude`; entry all-checks 25% → 50%. Logged #13 topic drift, #14 lookup mid check-in invented, #15 moves on after hard news. NOTES M7 steps 4–7.

## Decision log

| Decision | Why | Where |
|---|---|---|
| Default llama3.1:8b; qwen3 dropped 2026-10-22 | qwen3 CJK leaks (M1); missed 40% of tool calls, date args 43% vs llama 90% (M4) | `core/config.py`, NOTES M1/M4 |
| All schema fields required, lists capped | Optional fields skipped; unbounded list looped forever | `models/entry.py` |
| Field descriptions injected into prompt | Grammar doesn't pass descriptions to the model | `field_guide` |
| Guardrails in code for gratitude and mood_words | Prompt fix v1.4 regressed another case | `apply_guardrails` |
| Full-stack layout, uv workspace | User wanted traditional layers; `uv run` works from root | root + `backend/pyproject.toml` |
| Rule-based router, no LLM fallback yet | Cheap, testable; add LLM fallback once M7 can measure it | `nodes.router` |
| Deps in `Runtime` context, not state | Not serializable, shouldn't be checkpointed | `DiaryContext` |
| Separate `save` node after `confirm` | `interrupt()` reruns the node on resume | `confirm` / `save` |
| thread_id = date | One resumable conversation per day = one entry per day | `thread_config` |
| `check_same_thread=False` on SQLite | Streaming runs sync nodes in worker threads | `entry_repository.connect` |
| Only read-only tools; no `save_entry` tool | Model must never write without a human yes | `tools.py` |
| Code gate (`LOOKUP_REQUEST`) decides if tools are offered | Prompt rule left false calls at 75%; gate → 0% | `chat_agent` |
| Date cheat-sheet instead of letting model compute dates | 8B model got "Monday this week" wrong | `calendar_hint`, v1.7 |
| `search_entries` has only `query` | Optional date args filled with junk | `tools.py` docstring |
| Cosine cutoff 0.43 | Relevant ≤ 0.41, unrelated ≥ 0.45 on 10-entry seed; kNN invents matches otherwise | `MAX_DISTANCE` |
| Chroma = index, SQLite = truth; embed extracted fields | Index rebuildable; transcript chit-chat is noise | `vector_repository` |
| Summaries via CLI, not a chat reply | A review in chat lands in today's transcript → extraction copies past events | `nodes.summarize`, `cli summary` |
| Calendar periods (Mon–Sun, month), `--date` picks one | Matches `calendar_hint`; saved records keyed by `(period, start_date)` | `period_range` |
| Map step in code from approved fields, not an LLM | Extraction already condensed each day and the user approved it; the LLM map was a second, unreviewed copy (pottery distortion) | `day_line` |
| No stored `digest` column | Tried 2026-10-26: verbatim short entries lost context; summary + fields does the job | NOTES M5 |
| Collapse only over budget | A month of day lines fits (~2k tokens for 31 days); merging loses dates | `collapse` |
| Retry loop shared (`invoke_structured`) | Same structured-output retry for entries and summaries | `services/extraction.py` |
| Computed trend line removed | Measurably hurt the review (low/high named 2/6 vs 6/6) | `stats_text` docstring |
| Code writes mood_trend for < 3 rated days | Prompt rule failed 0/3; 2 points aren't a trend | `apply_guardrails` (summary) |
| CI runs fake-backed tests on Windows too + privacy guard | M4 hit Windows-only bugs; diary data must never be tracked | `.github/workflows/ci.yml` |
| Memories in system prompt, not `messages` | Messages become transcript → entry | `_companion_prompt` |
| Every memory write confirmed; `remember_fact` only proposes | Pollution compounds across all future prompts | `confirm_memory` |
| Staging in state instead of `interrupt()` inside the tool | ToolNode runs parallel calls in a thread pool | `remember_fact`, `merge_pending` |
| DUPLICATE 0.90 / RELATED 0.70 same-kind | Similarity can't tell update from different fact; prefer a 2nd memory over overwriting | `services/memory.py` |
| Code drops assistant reply after a lookup (not a "today only" prompt rule) | A/B 72% vs 71%; rule pulled assistant words back in | `format_transcript` |
| Eval datasets as JSONL labelled with correct answers | Known misses must show as failures | `evals/datasets/` |
| Custom tool checks, no `agentevals`; own judge, no `openevals` | Bugs were in tool args; 8B judge needs `invoke_structured` retry | `evaluators.check_tool_turn`, `judge.py` |
| Judge = llama3.1:8b, pinned apart from the agent (2026-10-27) | Ruler must not move when the agent migrates; self-preference risk accepted, measured by calibration | `judge.JUDGE_MODEL` |
| Judge checks count only after calibration (agree ≥ 85%, fail_catch ≥ 80%) | Uncalibrated r1 grounded agreed 38% | `judge.CALIBRATED` |
| An answer to a gratitude question counts as explicit gratitude | Guardrail emptied real gratitude in 5/12 simulated check-ins | `extraction.asked_for_gratitude` |

## Open questions / coming next
- Judge: r4 for grounded + follow_up, tested on a fresh `add-sim` batch (not rows 29–40); `kind` unverified; `CALIBRATED` empty. M8 optional polish.
- Open issues: #9 past-day leak (also user-mentioned past events, 2/2), #10 non-events in highlights, #11 router plain-language misses, #12 search cutoff margin, #13 check-in topic drift, #14 lookups mid check-in invented, #15 moves on after hard news.
- Known M7 findings to fix: router misfires ("check in at the hotel", plain-word "save it"), retrieval cutoff margin 0, extraction `not_in`/`forbidden` 50% (assistant's words, plans as events, "8 out of 10", wrong `work` tag), issue #9.
- Known M6 limits: value changes become a 2nd memory; declined candidate re-proposed same day; chat-only days add nothing; history trimming not done (full day resent).
- Known M5 limits: overview still overgeneralizes from sparse data ("great month" from 1 entry; sparse check 2/3); keyword "grounded" check is crude; saved summaries go stale if entries change; chars/4 underestimates CJK.
- Known M4 limits: `LOOKUP_REQUEST` needs "?" or a leading request verb (M7: "How are you?" offers tools).
- Known M3 limits: keyword router misfires; full history resent each turn (M6); past-midnight chat stays on yesterday's thread.

## Learner progress
| Topic | Status (solid / shaky / not yet) | Evidence / date |
|---|---|---|
| Required fields + grammar (optional = skipped) | solid | M2 quiz Q2 correct, 2026-09-28 |
| Bounded lists / runaway generation | solid-ish | M2 quiz Q4: got max_length; missed num_predict backstop, 2026-09-28 |
| Two-layer schema (ExtractedEntry vs JournalEntry) | shaky | M2 quiz Q1: said "mood" is code-known, 2026-09-28 |
| Grammar doesn't pass descriptions → field_guide | shaky | M2 quiz Q3: guessed "second validation", 2026-09-28 |
| Guardrails in code vs prompt rules | solid-ish | M2 Q5 "no idea" (2026-09-28) → M4 Q2 "move logic to code" ✓, but mixed in the date-arg removal; didn't name LOOKUP_REQUEST, 2026-10-24 |
| State vs context (M3) | shaky | M3 quiz Q1/Q2, 2026-10-22. Teach via "can it be written to a DB row?" |
| interrupt() reruns node from top (M3) | shaky | M3 quiz Q3: missed that it runs *twice*, 2026-10-22 |
| thread_id = date trade-offs (M3) | shaky | M3 quiz Q4, 2026-10-22 |
| Streaming filter / worker-thread SQLite bug (M3) | not yet | M3 quiz Q5: "no idea", 2026-10-22 |
| Tool msgs kept out of transcript (M4) | solid | M4 Q3 correct, 2026-10-24 |
| SQLite = truth, Chroma = rebuildable index (M4) | shaky | M4 Q1: only said "id = date", missed get_entry from SQLite + reindex, 2026-10-24 |
| No save tool = HITL via interrupt() (M4/M3) | shaky | M4 Q4: said "save node exists"; didn't connect to interrupt()/human yes, 2026-10-24 |
| Tool errors as strings → model retries (M4) | shaky | M4 Q5: saw parse fails; missed error ToolMessage → retry with calendar_hint → MAX_TOOL_ROUNDS, 2026-10-24 |
| Pattern across quizzes | — | Spots *what* breaks, skips *which mechanism* handles it. Ask "what code path runs next?" |
