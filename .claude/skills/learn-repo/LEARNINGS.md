# Repo learnings (AI-maintained)

last_synced_commit: 2253cab
last_synced: 2026-10-20
current_milestone: M3 done, M4 (tools / ReAct) next

> Maintained by the `learn-repo` skill. The code map and concept index live here. The narrative, experiments, and eval tables live in `docs/NOTES.md`. Roadmap lives in `docs/PLAN.md`.

## End-to-end flow (as of M3)

Layers: `cli/` → `services/` → `repositories/` → `models/`; `core/` (config, llm, prompts) is used everywhere. Paths below are under `backend/src/diary_agent/`.

```
uv run diary chat  (cli/app.py → chat())
  ├─ DiaryContext(chat_model, extractor, conn, day, model_name)   deps, passed via context=, never checkpointed
  ├─ build_graph(open_checkpointer())   SqliteSaver → data/checkpoints.db
  ├─ thread_config(day)                 thread_id = today's date → resume after restart
  └─ each user line → run_turn → stream_turn(graph, {"user_input": text}) = ONE graph run
       START → router (rule-based: /save /summary /cancel /checkin, "check in" keyword, mode)
         ├─ chat_agent: COMPANION_SYSTEM + all messages → chat_model  → END
         ├─ checkin:    + CHECKIN_SYSTEM(topic = CHECKIN_TOPICS[step]), step+1 → END
         ├─ summarize:  stub notice (M5) → END
         ├─ done (/cancel) → END
         └─ extract: extract_entry (M2 pipeline, unchanged) → draft as JSON dict
              └─ confirm: interrupt({draft, exists}) ── graph pauses + checkpoints
                   CLI: pending_interrupt → ask_to_save → run_turn(Command(resume=True/False))
                   ├─ True  → save: JournalEntry(**draft, code-known fields) → entry_repo.save_entry → END
                   └─ False → END ("Not saved.")
       stream_mode=["messages","updates"]: tokens only from REPLY_NODES; "updates" → status line
  exit → unsaved (len(messages) > saved_count)? → offer /save
uv run diary list / show → entry_repo.list_entries / get_entry
```

## File map

| File | What it does | Key symbols | Concepts it teaches |
|---|---|---|---|
| `backend/src/diary_agent/core/config.py` | Env/.env-driven frozen settings | `Settings`, `settings`, `PROJECT_ROOT` | Config in one place; model choice is config, not code |
| `backend/src/diary_agent/core/llm.py` | Model factories | `get_chat_model(temperature, max_tokens)`, `get_embeddings()` | Depend on `BaseChatModel` interface; provider swap = one file; `reasoning` flag only sent when on |
| `backend/src/diary_agent/core/prompts.py` | System prompts + version | `PROMPT_VERSION` (v1.5), `COMPANION_SYSTEM`, `CHECKIN_SYSTEM`, `EXTRACT_SYSTEM` | Prompt versioning so eval runs are comparable; CHECKIN_SYSTEM is appended, code picks the topic |
| `backend/src/diary_agent/services/agent/state.py` | Graph state + run deps | `DiaryState(MessagesState)`, `DiaryContext`, `Route` | State (checkpointed, serializable) vs context (deps); `add_messages` reducer appends, other keys overwrite |
| `backend/src/diary_agent/services/agent/nodes.py` | Node functions | `router`, `route_after_router`, `chat_agent`, `checkin`, `summarize`, `extract`, `route_after_extract`, `confirm`, `save`, `CHECKIN_TOPICS` | Nodes return partial updates; deterministic router; `interrupt()` + `Command(goto=)`; write split out of the interrupting node |
| `backend/src/diary_agent/services/agent/graph.py` | Wiring + runtime helpers | `build_graph`, `open_checkpointer`, `thread_config`, `stream_turn`, `pending_interrupt`, `REPLY_NODES` | `StateGraph`, conditional edges, `SqliteSaver`, thread_id, dual stream modes, `get_state().interrupts` |
| `backend/src/diary_agent/models/entry.py` | Pydantic schemas | `ExtractedEntry` (LLM-inferred), `JournalEntry` (+ code-known fields) | Two-layer schema; all fields required; bounded lists; field order = evidence before judgement |
| `backend/src/diary_agent/services/extraction.py` | Transcript → entry | `build_extractor`, `extract_entry`, `apply_guardrails`, `field_guide`, `format_transcript`, `ExtractionError` | Structured output, retry-with-error-feedback, deterministic guardrails, schema as single source of truth for prompt |
| `backend/src/diary_agent/repositories/entry_repository.py` | SQLite repo | `connect` (`check_same_thread=False`), `save_entry` (INSERT OR REPLACE), `get_entry`, `list_entries`, `LIST_FIELDS` | Lists stored as JSON text; pydantic round-trip via `model_validate` |
| `backend/src/diary_agent/cli/app.py` | Typer CLI, thin driver of the graph | `chat`, `run_turn`, `ask_to_save`, `list_cmd`, `show`, `render_entry` | CLI answers interrupts and resumes; resume-on-start (incl. a paused interrupt); BOM strip for PowerShell piping |
| `scripts/smoke.py` | Ollama connectivity check | — | `.stream()`, embedding dim (768) |
| `evals/extract_probe.py` | M2 mini eval: 7 cases × N runs | `CASES`, `check()` | Deterministic checks, mood *ranges*, `forbidden` words, repeated runs for nondeterminism. Seed of the M7 harness |
| `backend/tests/services/test_extraction.py` | Retry + guardrail tests | `scripted()` fake extractor via `RunnableLambda` | Test LLM control flow with scripted fakes |
| `backend/tests/services/agent/test_router.py` | Router unit tests | parametrized routes | A deterministic router is testable without any model |
| `backend/tests/services/agent/test_graph.py` | Graph behaviour tests | restart, separate days, interrupt approve/decline, check-in, `stream_turn` | Fake models + temp checkpointer; regression test runs through `stream_turn` (the CLI's path) |
| `backend/tests/repositories/test_entry_repository.py` | SQLite round-trip tests | — | Repo layer tested in isolation |

## Concept index

### M0: Models and messages
- **Chat model + messages:** `SystemMessage`/`HumanMessage`/`AIMessage` in, `AIMessage` out. `core/llm.py`, `scripts/smoke.py`.
- **invoke vs stream:** whole reply vs chunks. `ChatSession.stream`.
- **Embeddings:** text → 768-dim vector (nomic-embed-text), unused until M4 RAG. `get_embeddings()`.

### M1: LCEL and history
- **LCEL / Runnables:** `prompt | model | parser`. Every Runnable gets invoke/stream/batch. `chat.build_chain`.
- **Stateless model:** "memory" = resending history each call. M1 kept it in `ChatSession.history` (deleted in M3); now it lives in `DiaryState.messages`. Still resent in full each turn; trimming comes in M6.
- **Tests vs evals:** unit tests check plumbing with fakes. Evals measure real model quality. See NOTES.md → M1.
- **First eval (CJK leak):** fixed inputs + regex checker + repeats → switched default to llama3.1:8b. NOTES.md → M1 "Observed failure".

### M2: Structured output and guardrails
- **`with_structured_output(method="json_schema", include_raw=True)`:** returns `{raw, parsed, parsing_error}`, which enables retry with feedback. `extract.build_extractor`.
- **Grammar-constrained decoding (Ollama):** guarantees structure, not content. Only *required* fields are forced, and descriptions aren't seen by the model. `models/entry.py` docstring.
- **Retry with error feedback:** show the model its bad output + the validation error. `extract_entry`.
- **Deterministic guardrails > prompt tweaks** for rules checkable in code. They can't regress other cases. `apply_guardrails`.
- **Never ask the model for what code knows:** date, transcript, model, and version come from code. `JournalEntry`.
- **Human-in-the-loop writes:** confirm save and overwrite. `cli.save_flow`.
- **Eval hygiene:** rerun the whole suite after every change; sometimes the label is wrong; temp 0 isn't deterministic; 100% = eval too easy. NOTES.md → M2 "eval loop".

### M3: LangGraph (state, nodes, edges, checkpoints, interrupts)
- **StateGraph = nodes + edges over a shared state.** A node is `(state, runtime) -> partial update`. `graph.build_graph`.
- **State vs context:** `DiaryState` = data, checkpointed after every node, must serialize (so `draft` is a dict, not a pydantic model). `DiaryContext` = deps, passed fresh via `context=`, read as `runtime.context`. Rule: survives a restart → state; a tool you use → context.
- **Reducers:** `messages` uses `add_messages` (append); other keys are last-write-wins.
- **Conditional edge vs `Command`:** router writes `route`, edge reads it (routing visible in graph.py). `confirm` returns `Command(goto=…, update=…)` (decide + update in one step).
- **Multi-turn flow without a loop:** one invoke = one user turn; `mode` + `checkin_step` in state tell the next turn's router where we are. Code picks the topic, the model words it.
- **Commands stay out of `messages`:** `/save` never pollutes the transcript the extractor reads; status goes through `notice`.
- **Checkpointer = short-term memory:** `SqliteSaver`, `thread_id` = date. Restart resumes; tomorrow is a new thread. `get_state_history` shows every step.
- **`interrupt()` = HITL:** pauses + checkpoints; resume with `Command(resume=value)`, which becomes interrupt()'s return. **Node reruns from the top on resume**, so only idempotent work before it (DB read OK; write is in `save`).
- **Streaming:** `stream_mode=["messages","updates"]`; filter tokens by `meta["langgraph_node"]` so extraction JSON isn't printed.
- **Test the path users run:** worker-thread SQLite crash passed `invoke` tests, failed in `stream`. NOTES.md → M3 "Bug found by running".

## Decision log

| Decision | Why | Where |
|---|---|---|
| Default model llama3.1:8b over qwen3:8b | qwen3 leaked CJK in 1–3/9 replies; llama 0/27 | `core/config.py`, NOTES M1 |
| All schema fields required, lists capped | Optional fields skipped; unbounded list looped forever | `models/entry.py` |
| Field descriptions injected into prompt | Grammar doesn't pass descriptions to the model | `services/extraction.field_guide` |
| Evidence fields before `mood` | Model writes JSON top-down; judge after facts | `ExtractedEntry` field order |
| Extraction at temperature 0, `num_predict=512` | Most likely reading; runaway safety cap | `get_extraction_model` |
| Guardrails in code for gratitude and mood_words | Prompt fix v1.4 regressed another case | `apply_guardrails` |
| Full-stack layout: `backend/` (core / models / repositories / services / cli / api) + `frontend/` placeholder; root is a uv workspace | User asked for a traditional full-stack structure with data models kept separate. Layer rule: cli/api → services → repositories → models, and core is usable everywhere. The workspace keeps `uv run …` working from the root | root + `backend/pyproject.toml`, `docs/PLAN.md` |
| Rule-based router, no LLM fallback yet | Cheap, instant, testable; add LLM fallback only once M7 routing evals can measure it ("check in at the hotel" misfires) | `nodes.router`, PLAN M3 |
| Deps in `Runtime` context, not state | Models / DB connections aren't serializable and shouldn't be checkpointed | `state.DiaryContext` |
| Separate `save` node after `confirm` | `interrupt()` reruns the node on resume; keep side effects out of it | `nodes.confirm` / `nodes.save` |
| thread_id = date | One resumable conversation per day, matches one entry per day | `graph.thread_config` |
| `check_same_thread=False` on SQLite | LangGraph runs sync nodes in worker threads when streaming; one node at a time, so safe | `entry_repository.connect`, `graph.open_checkpointer` |
| One entry per day, confirm overwrite | Simple model; human approves destructive writes | `repositories/entry_repository.py`, `cli/app.save_flow` |

## Open questions / coming next
- M4: `chat_agent` becomes a tool-calling agent (`ToolNode` + `tools_condition`) with `search_entries` (Chroma), `get_entries_by_date`, `save_entry`; entries embedded on save. Motivated by the M3 hallucinated "laughter" detail (NOTES M3).
- Known M3 limits: keyword router misfires; full history resent each turn (M6); a chat past midnight stays on yesterday's thread until restart.
- M4: will qwen3 beat llama3.1 at tool calling enough to justify per-task models? Decide with evals.
- Probe gaps: "8 out of 10" became a highlight, and a `work` tag landed on a dentist trip. No checks on tags or highlights yet (M7).

## Learner progress
| Topic | Status (solid / shaky / not yet) | Evidence / date |
|---|---|---|
| Required fields + grammar (optional = skipped) | solid | M2 quiz Q2 correct, 2026-09-28 |
| Bounded lists / runaway generation | solid-ish | M2 quiz Q4: got max_length; didn't mention num_predict as backstop, 2026-09-28 |
| Two-layer schema (ExtractedEntry vs JournalEntry) | shaky | M2 quiz Q1: said "mood" (it's LLM-inferred); answer is date/transcript/model/prompt_version, 2026-09-28 |
| Grammar doesn't pass descriptions → field_guide | shaky | M2 quiz Q3: guessed "second validation", 2026-09-28 |
| Guardrails in code vs prompt rules | not yet | M2 quiz Q5: "no idea", 2026-09-28. Revisit with the v1.4 regression story |
| State vs context (M3) | shaky | M3 quiz Q1: put chat model in state; got messages right. Q2: didn't connect dict draft to checkpoint serialization, 2026-10-22. Teach via "can it be written to a DB row?" |
| interrupt() reruns node from top (M3) | shaky | M3 quiz Q3: saw side effect runs pre-approval, missed that it runs *twice* on resume, 2026-10-22 |
| thread_id = date trade-offs (M3) | shaky | M3 quiz Q4: named one-entry-per-day; missed resume-after-restart + past-midnight limit, 2026-10-22 |
| Streaming filter / worker-thread SQLite bug (M3) | not yet | M3 quiz Q5: "no idea", didn't know REPLY_NODES filter exists, 2026-10-22 |
