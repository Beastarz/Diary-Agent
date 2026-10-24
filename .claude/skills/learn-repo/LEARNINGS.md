# Repo learnings (AI-maintained)

last_synced_commit: a30ecf3
last_synced: 2026-10-24
current_milestone: M5 done, M6 (long-term memory) next

> Maintained by the `learn-repo` skill. The code map and concept index live here. The narrative, experiments, and eval tables live in `docs/NOTES.md`. Roadmap lives in `docs/PLAN.md`.

## End-to-end flow (as of M5)

Layers: `cli/` → `services/` → `repositories/` → `models/`; `core/` (config, llm, prompts) is used everywhere. Paths below are under `backend/src/diary_agent/`. The full ASCII graph is drawn in the `graph.py` module docstring.

```
uv run diary chat  (cli/app.py → chat())
  ├─ DiaryContext(chat_model, extractor, conn, vectors, day, model_name)   deps, via context=, never checkpointed
  ├─ build_graph(open_checkpointer())   SqliteSaver → data/checkpoints.db
  ├─ thread_config(day)                 thread_id = today's date → resume after restart
  └─ each user line → run_turn → stream_turn(graph, {"user_input": text}) = ONE graph run
       START → router (rule-based: /save /summary /cancel /checkin, "check in" keyword, mode)
         ├─ chat_agent ⇄ tools   (ReAct loop, M4)
         │    LOOKUP_REQUEST gate: no "?" / request verb → plain invoke, no tools → END
         │    else COMPANION_SYSTEM + TOOLS_SYSTEM(calendar_hint) → chat_model.bind_tools(TOOLS)
         │    tools_condition: tool_calls? → ToolNode(TOOLS) → ToolMessages → back to chat_agent
         │    no tool_calls (or MAX_TOOL_ROUNDS=3 used, tool_calls dropped) → END
         │    tools: search_entries (Chroma, cutoff 0.43 → dates → SQLite) | get_entries_by_date (SQLite)
         ├─ checkin:    + CHECKIN_SYSTEM(topic = CHECKIN_TOPICS[step]), step+1 → END   (no tools)
         ├─ summarize:  notice "run 'diary summary'" → END   (kept out of chat: would pollute today's transcript)
         ├─ done (/cancel) → END
         └─ extract: format_transcript (drops tool msgs) → extract_entry → draft dict
              └─ confirm: interrupt({draft, exists}) ── pause + checkpoint
                   CLI: pending_interrupt → ask_to_save → run_turn(Command(resume=True/False))
                   ├─ True  → save: SQLite save_entry, then vector_repo.upsert_entry (failure → "run reindex") → END
                   └─ False → END ("Not saved.")
       stream_mode=["messages","updates"]: tokens only from REPLY_NODES; updates → status
         ("Writing up your entry...", "Searching your journal...")
  exit → unsaved (len(messages) > saved_count)? → offer /save
uv run diary list / show / reindex (rebuild Chroma from SQLite)
uv run diary summary --week|--month [--date D] [--saved]   (cli/app.py → summary(), no graph)
  period_range (calendar Mon–Sun / month) → list_entries(start, end) → none? stop, no LLM call
  summarize_period(writer, reducer, entries, ...)   all calls num_ctx=8192, temperature 0
    map:      condense → writer.batch([CONDENSE_SYSTEM + entry_text(e)])   entry_text = fields + User: lines, capped
              code prefixes each digest "Mon 2026-10-19 (mood 4): ..."
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
| `core/prompts.py` | System prompts + version | `PROMPT_VERSION` (v1.8), `COMPANION_SYSTEM`, `CHECKIN_SYSTEM`, `TOOLS_SYSTEM`, `EXTRACT_SYSTEM`, `CONDENSE_SYSTEM`, `COLLAPSE_SYSTEM`, `REDUCE_SYSTEM` | Prompt versioning; TOOLS_SYSTEM appended only when tools are bound; `{calendar}` filled by code |
| `services/agent/state.py` | Graph state + run deps | `DiaryState(MessagesState)`, `DiaryContext` (+ `vectors`), `Route` | State (checkpointed) vs context (deps); `add_messages` appends |
| `services/agent/nodes.py` | Node functions | `router`, `chat_agent`, `LOOKUP_REQUEST`, `calendar_hint`, `_tool_rounds_this_turn`, `MAX_TOOL_ROUNDS`, `GAVE_UP`, `checkin`, `extract`, `confirm`, `save` | Code gate on *whether* tools are offered; code does date math; loop budget; save indexes after SQLite |
| `services/agent/tools.py` | Read-only tools | `search_entries`, `get_entries_by_date`, `format_entry`, `_parse_range`, `MAX_DISTANCE`, `SEARCH_K`, `MAX_LISTED`, `TOOLS` | `@tool` docstring + `Annotated` args = the whole schema; `ToolRuntime[DiaryContext]` hidden from model; errors as strings; distance cutoff; capped output |
| `services/agent/graph.py` | Wiring + runtime helpers | `build_graph` (ToolNode, `tools_condition`), `stream_turn`, `pending_interrupt`, `REPLY_NODES` | Loop = conditional edge + edge back; `isinstance(chunk, AIMessage)` (chunks are a subclass) |
| `repositories/vector_repository.py` | Chroma index, one vector per day | `open_store` (cosine, telemetry off), `entry_text`, `upsert_entry` (id = date), `search` → `(date, distance)`, `_day_key`, `DOC_PREFIX`/`QUERY_PREFIX`, `count` | Vector store as a rebuildable index, SQLite is truth; embed extracted fields, not transcript; nomic task prefixes; numeric date metadata for `$gte` |
| `models/entry.py` | Pydantic schemas | `ExtractedEntry` (LLM-inferred), `JournalEntry` (+ code-known fields) | Two-layer schema; all fields required; bounded lists; evidence before judgement |
| `services/extraction.py` | Transcript → entry | `build_extractor`, `extract_entry`, `invoke_structured` (shared retry loop), `apply_guardrails`, `field_guide(schema)`, `format_transcript` | Structured output, retry-with-feedback, guardrails; transcript = only human/ai with content |
| `models/summary.py` | Summary schemas | `PeriodSummary` (LLM-written: themes, wins, challenges, mood_trend, overview), `SummaryRecord` (+ range, count, `mood_avg`, model, version), `Period` | Same two-layer pattern as entry.py; numbers are code-known |
| `services/summary.py` | Entries → review, map-reduce | `summarize_period`, `condense`, `collapse`, `_groups`, `reduce`, `apply_guardrails`, `stats_text`, `entry_text`, `period_range`, `mood_avg`, `estimate_tokens`, `build_writer`/`build_reducer`/`get_summary_model`, `NUM_CTX`, `MAP_INPUT_TOKENS`, `REDUCE_INPUT_TOKENS`, `MAX_COLLAPSE_ROUNDS`, `MIN_TREND_DAYS` | `batch()` = map; token budget (chars/4); collapse as last resort; code computes stats; guardrails override sparse trend |
| `repositories/entry_repository.py` | SQLite repo | `connect` (`check_same_thread=False`), `save_entry`, `get_entry`, `list_entries(start, end)` | Lists as JSON text; pydantic round-trip |
| `repositories/summary_repository.py` | SQLite summaries in journal.db | `save_summary`, `get_summary(period, start)`, `ensure_schema`; PK `(period, start_date)` | Reuses `entry_repository.connect`; table created on first use |
| `cli/app.py` | Typer CLI, thin graph driver | `chat`, `run_turn`, `ask_to_save`, `list_cmd`, `show`, `reindex`, `summary` (`--week/--month` one bool flag), `render_summary` | CLI answers interrupts; resume-on-start; `reindex` repairs the index; summary save needs a yes |
| `evals/extract_probe.py` | M2 extraction eval, 7 cases × N | `CASES`, `check()` | Deterministic checks, mood ranges, repeats |
| `evals/tool_probe.py` | M4 tool-use eval, 14 cases × N, real graph | `TODAY` (fixed Thursday), `SEED` (10 entries), checks tool/args/answer/clean + model-free retrieval | Split missed-call vs false-call rates; separate retrieval from reasoning; results JSON tagged model + prompt + commit |
| `evals/summary_probe.py` | M5 summary probe, 3 cases × N via real `summarize_period` | reuses tool_probe `SEED`; `spy()` counts map/collapse/reduce by system prompt; checks grounded/mood/recurring/embellish/sparse/clean; `--collapse` | Inspect intermediate outputs (digests), not just the final answer |
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
- **Map-reduce:** map = condense each entry (`writer.batch()`, LCEL's map); collapse only when over budget; reduce = structured `PeriodSummary`. Every call has a bounded input. `summarize_period`.
- **Context window is a budget:** Ollama's default window cuts overflow *silently*. One fixed `NUM_CTX = 8192` for all summary calls (changing it reloads the model). Tokens estimated as chars/4.
- **Cut cheap things first:** map input keeps only `User:` lines (assistant words aren't facts, ~half the tokens). `entry_text`.
- **Code computes the numbers** (avg, low/high day, recurring tags) → `stats_text`; code writes the digest date/mood prefix.
- **But more computed facts aren't free:** a "Mood trend: down" line made the review skip low/high days (2/6 vs 6/6) → removed.
- **Length demands cause invention:** "2-3 sentences" padded a short entry ("first project is now complete") → `CONDENSE_SYSTEM` "at most 3, one is fine".
- **Prompt rule failed again (0/3)** on sparse "steady throughout" → `apply_guardrails` writes `mood_trend` when < `MIN_TREND_DAYS`; also drops "good mood" as a win.
- **Collapse has a floor:** forced 150-token budget → 9 extra calls, 2.3× slower, still over budget after 3 rounds; dates lost. `MAX_COLLAPSE_ROUNDS`.
- **Errors compound across hops:** the pottery distortion came from *map*; reduce repeated it. Probe saves digests. NOTES M5.

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
| Map always runs; collapse only over budget | Digests make a month fit (~1.4k tokens for 31 days); merging loses dates | `summarize_period`, `collapse` |
| Retry loop shared (`invoke_structured`) | Same structured-output retry for entries and summaries | `services/extraction.py` |
| Computed trend line removed | Measurably hurt the review (low/high named 2/6 vs 6/6) | `stats_text` docstring |
| Code writes mood_trend for < 3 rated days | Prompt rule failed 0/3; 2 points aren't a trend | `apply_guardrails` (summary) |
| CI runs fake-backed tests on Windows too + privacy guard | M4 hit Windows-only bugs; diary data must never be tracked | `.github/workflows/ci.yml` |

## Open questions / coming next
- M6: long-term memory (LangGraph store, `update_memory` node, `remember_fact` tool, history trimming).
- Known M5 limits: overview still overgeneralizes from sparse data ("great month" from 1 entry; sparse check 2/3); keyword "grounded" check is crude; saved summaries go stale if entries change; chars/4 underestimates CJK.
- Known M4 limits: `LOOKUP_REQUEST` needs "?" or a leading request verb; `MAX_DISTANCE` calibrated on only 10 synthetic entries; the assistant's reply about a past day stays in today's transcript (extraction could pick it up — watch in M7).
- Known M3 limits: keyword router misfires; full history resent each turn (M6); past-midnight chat stays on yesterday's thread.
- M7: LLM judge to replace keyword checkers (several "failures" were checker misses); checks on tags/highlights.

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
