# Repo learnings (AI-maintained)

last_synced_commit: 01f32ce
last_synced: 2026-09-28
current_milestone: M2 done, M3 (LangGraph) next

> Maintained by the `learn-repo` skill. The code map and concept index live here. The narrative, experiments, and eval tables live in `docs/NOTES.md`. Roadmap lives in `docs/PLAN.md`.

## End-to-end flow (as of M2)

Layers: `cli/` → `services/` → `repositories/` → `models/`; `core/` (config, llm, prompts) is used everywhere. Paths below are under `backend/src/diary_agent/`.

```
uv run diary chat  (cli/app.py → chat())
  └─ ChatSession (services/chat.py)          in-memory history list, lost on exit
       └─ build_chain: ChatPromptTemplate(COMPANION_SYSTEM + history + input) | ChatOllama | StrOutputParser
            └─ .stream() → chunks printed live
  /save or exit → save_flow (cli/app.py)
       ├─ format_transcript(history) → "User: … / Assistant: …" text
       ├─ extract_entry (services/extraction.py)
       │    ├─ model.with_structured_output(ExtractedEntry, json_schema, include_raw=True), temp 0, 512 tokens
       │    ├─ on parse failure: append raw output + error, retry (max 2 retries) → ExtractionError
       │    └─ apply_guardrails: drop non-explicit gratitude, drop mood_words not in user text
       ├─ render + Confirm (human approves every write, and any overwrite)
       └─ JournalEntry(**extracted, date, transcript, model, prompt_version) → entry_repo.save_entry (repositories/entry_repository.py, SQLite, 1 row/day)
uv run diary list / show → entry_repo.list_entries / get_entry
```

## File map

| File | What it does | Key symbols | Concepts it teaches |
|---|---|---|---|
| `backend/src/diary_agent/core/config.py` | Env/.env-driven frozen settings | `Settings`, `settings`, `PROJECT_ROOT` | Config in one place; model choice is config, not code |
| `backend/src/diary_agent/core/llm.py` | Model factories | `get_chat_model(temperature, max_tokens)`, `get_embeddings()` | Depend on `BaseChatModel` interface; provider swap = one file; `reasoning` flag only sent when on |
| `backend/src/diary_agent/core/prompts.py` | System prompts + version | `PROMPT_VERSION` (v1.4), `COMPANION_SYSTEM`, `EXTRACT_SYSTEM` | Prompt versioning so eval runs are comparable; changelog comments |
| `backend/src/diary_agent/services/chat.py` | M1 LCEL chat | `build_chain()`, `ChatSession.stream/has_unsaved/mark_saved` | LCEL `|`, `MessagesPlaceholder`, stateless models → you manage history. Superseded in M3 |
| `backend/src/diary_agent/models/entry.py` | Pydantic schemas | `ExtractedEntry` (LLM-inferred), `JournalEntry` (+ code-known fields) | Two-layer schema; all fields required; bounded lists; field order = evidence before judgement |
| `backend/src/diary_agent/services/extraction.py` | Transcript → entry | `build_extractor`, `extract_entry`, `apply_guardrails`, `field_guide`, `format_transcript`, `ExtractionError` | Structured output, retry-with-error-feedback, deterministic guardrails, schema as single source of truth for prompt |
| `backend/src/diary_agent/repositories/entry_repository.py` | SQLite repo | `connect`, `save_entry` (INSERT OR REPLACE), `get_entry`, `list_entries`, `LIST_FIELDS` | Lists stored as JSON text; pydantic round-trip via `model_validate` |
| `backend/src/diary_agent/cli/app.py` | Typer CLI | `chat`, `save_flow`, `list_cmd`, `show`, `render_entry` | Human-in-the-loop confirmation before writes; BOM strip for PowerShell piping |
| `scripts/smoke.py` | Ollama connectivity check | — | `.stream()`, embedding dim (768) |
| `evals/extract_probe.py` | M2 mini eval: 7 cases × N runs | `CASES`, `check()` | Deterministic checks, mood *ranges*, `forbidden` words, repeated runs for nondeterminism. Seed of the M7 harness |
| `backend/tests/services/test_chat.py` | Chat plumbing tests | — | `FakeListChatModel` → tests without Ollama |
| `backend/tests/services/test_extraction.py` | Retry + guardrail tests | `scripted()` fake extractor via `RunnableLambda` | Test LLM control flow with scripted fakes |
| `backend/tests/repositories/test_entry_repository.py` | SQLite round-trip tests | — | Repo layer tested in isolation |

## Concept index

### M0: Models and messages
- **Chat model + messages:** `SystemMessage`/`HumanMessage`/`AIMessage` in, `AIMessage` out. `core/llm.py`, `scripts/smoke.py`.
- **invoke vs stream:** whole reply vs chunks. `ChatSession.stream`.
- **Embeddings:** text → 768-dim vector (nomic-embed-text), unused until M4 RAG. `get_embeddings()`.

### M1: LCEL and history
- **LCEL / Runnables:** `prompt | model | parser`. Every Runnable gets invoke/stream/batch. `chat.build_chain`.
- **Stateless model:** "memory" = resending history each call. `ChatSession.history`. Unbounded growth, fixed later (M3 checkpointer, trimming).
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
| One entry per day, confirm overwrite | Simple model; human approves destructive writes | `repositories/entry_repository.py`, `cli/app.save_flow` |

## Open questions / coming next
- M3: replace `ChatSession` with a LangGraph `StateGraph` (router → checkin / chat_agent / summarize), `SqliteSaver` checkpointer keyed by date, `interrupt()` for save confirmation.
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
