# Diary Agent — Local-First Chat Journal (learning build)

## Context
A chat-based daily journal agent, running entirely on free local models, built as a vehicle to learn LangChain, LangGraph, good agent design, and agent evals. Machine: Windows 11, Python 3.12, RTX 4060 (8 GB VRAM).

Decisions: **Terminal CLI**, **Ollama**, features = **guided check-in, search past entries, weekly/monthly summaries, long-term memory about you**, built in **staged milestones** (each runnable, each with a short `NOTES.md` section explaining the concepts learned).

## Status
| Milestone | Status |
|---|---|
| M0 Setup | ✅ done |
| M1 Plain LangChain chat | ✅ done |
| M2 Structured entries | ✅ done |
| M3 LangGraph conversion | ✅ done |
| M4 Tools / ReAct agent | ✅ done |
| M5 Summaries | ⏭ next |
| M6–M8 | todo |

Change log:
- **M1:** default chat model switched from `qwen3:8b` to `llama3.1:8b`. qwen3 leaked Chinese tokens into 1–3 of every 9 English replies; llama3.1 leaked in 0 of 27 (see NOTES.md). Revisit qwen3 for tool calling in M4 using evals.
- **M2:** one entry per day (re-save asks before overwriting). Ollama structured output needs *all fields required* and *bounded lists* (see NOTES.md M2). Added deterministic `apply_guardrails()` for gratitude and mood words. `evals/extract_probe.py` (7 cases) is the seed of the M7 extraction eval.
- **M3:** `services/chat.py` replaced by `services/agent/` (StateGraph + `SqliteSaver`, thread_id = date). The router is rule-based only (slash commands + "check in" keyword); the LLM fallback waits for M7 routing evals. Deps go through LangGraph's `Runtime` context, not the state. The check-in is 4 code-chosen topics across turns, then extract → `interrupt()` → save. SQLite connections need `check_same_thread=False` because LangGraph runs nodes in worker threads.
- **M4:** `chat_agent <-> tools` loop (`ToolNode` + `tools_condition`) with two read-only tools, `search_entries` (Chroma, cosine cutoff 0.43) and `get_entries_by_date`. There's **no `save_entry` tool**: writes stay behind `interrupt()`. `remember_fact` moves to M6. Entries are embedded on save; `diary reindex` rebuilds the index. `evals/tool_probe.py` drove four fixes: a date cheat-sheet in the prompt, no date args on search, a relevance cutoff, and a code gate that only offers tools for questions/requests (a prompt rule didn't stop llama's 75% false tool calls; the gate made it 0%). llama3.1 beat qwen3 on tool use (missed calls 7% vs 40%, date args 90% vs 43%), so it stays the default (see NOTES.md M4).
- **2026-10-22:** qwen3 dropped from the project (lost on CJK leaks in M1 and tool use in M4). llama3.1:8b is the only chat model; evals still take `--models` for future comparisons.
- **Structure (2026-09-28):** moved to a full-stack layout: `backend/` (layered `core / models / repositories / services / cli / api`) + `frontend/` placeholder, with the root as a uv workspace. See Project layout below.

## Stack
- **LLM**: Ollama `llama3.1:8b` (chose over qwen3:8b in M1 and again in M4; qwen3 since dropped). Model name configurable via `.env` (`DIARY_CHAT_MODEL`).
- **Embeddings**: Ollama `nomic-embed-text` (768 dims).
- **Libraries**: `langchain-core`, `langchain-ollama`, `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-chroma` (local vector store), `pydantic`, `rich` + `typer` (CLI), `pytest`, `openevals` / `agentevals` (to add in M7).
- **Packaging**: `uv` workspace. Root `pyproject.toml` holds dev tools + pytest config; `backend/pyproject.toml` is the `diary-agent` package with CLI entry point `diary = diary_agent.cli.app:app`.
- **Storage (all under `./data/`, git-ignored)**: `journal.db` (SQLite: entries table), `checkpoints.db` (LangGraph conversation state), `chroma/` (entry embeddings), `store.db` (long-term user memory).
- No cloud calls anywhere. LangSmith tracing is off by default; local tracing uses LangGraph's streaming plus a simple JSONL trace logger.

## Project layout
Traditional full-stack layout: `backend/` + `frontend/`, and a layered backend. `✓` = exists; the rest shows where later milestones' files go.

Layer rule: `cli/`, `api/` → `services/` → `repositories/` → `models/`. `core/` is usable from anywhere. Nothing imports "upward".

```
diary_agent/                     uv workspace root: pyproject.toml, uv.lock, .venv
  README.md
  backend/
    pyproject.toml               ✓ package `diary-agent`, `diary` CLI entry point
    .env.example                 ✓ copy to backend/.env
    src/diary_agent/
      core/                      infrastructure used by every layer
        config.py                ✓ settings from env/.env
        llm.py                   ✓ get_chat_model(), get_embeddings()
        prompts.py               ✓ system prompts + PROMPT_VERSION
      models/                    pydantic data models only (no logic, no I/O)
        entry.py                 ✓ ExtractedEntry / JournalEntry
        summary.py               M5
        memory.py                M6: user facts
      repositories/              data access; the only place with SQL / vector DB calls
        entry_repository.py      ✓ SQLite entries (connect, save_entry, get_entry, list_entries)
        vector_repository.py     ✓ Chroma index (upsert_entry, search, count)
        memory_repository.py     M6: LangGraph BaseStore wrapper
      services/                  business logic
        extraction.py            ✓ structured extraction + retries + guardrails
        summary.py               M5: map-reduce summaries
        agent/                   ✓ LangGraph agent (replaced M1's services/chat.py)
          state.py  nodes.py  graph.py   ✓ StateGraph + checkpointer + interrupt()
          tools.py               ✓ @tool search_entries, get_entries_by_date (remember_fact in M6)
      cli/
        app.py                   ✓ `diary chat` (/checkin, /save, /cancel), `list`, `show`, `reindex`; `summary` in M5
      api/                       HTTP API for the frontend (FastAPI, when a UI is built)
    tests/                       no Ollama; fakes; mirrors src/
      services/  services/agent/  repositories/   ✓
  frontend/                      placeholder for a future web UI
  evals/                         needs Ollama; measures model quality
    extract_probe.py             ✓ M2 mini eval (7 cases)
    tool_probe.py                ✓ M4 tool-use eval (14 cases, --models to compare, retrieval check)
    datasets/*.jsonl  evaluators.py  run_evals.py   M7
  scripts/
    smoke.py                     ✓ Ollama connectivity check
  docs/
    PLAN.md                      ✓ this file: roadmap + status
    NOTES.md                     ✓ per-milestone learning notes
```

## Milestones
Each milestone ends runnable. Append a docs/NOTES.md section (concepts, what to try, gotchas) and tick it off in README + this file.

**M0 — Setup. ✅** Ollama, uv, deps, `config.py`, `llm.py`, smoke script.
*Learn:* chat models, messages, streaming.

**M1 — Plain LangChain chat. ✅** `prompt | model` chain with a journaling-companion system prompt and in-memory history; CLI loop.
*Learn:* LCEL, prompt templates, why chat history must be managed explicitly.

**M2 — Structured entries.** `extract_entry` uses `model.with_structured_output(JournalEntry)` (pydantic: date, mood 1–10, mood_words, highlights, challenges, gratitude, tags, summary). Persist to SQLite via `repositories/entry_repository.py`.
*Learn:* structured output, schema design, validation/retry for small models.

**M3 — LangGraph conversion. ✅** Rebuild as a `StateGraph`: `router → {checkin | chat_agent | summarize}`. The guided check-in is a small sub-flow: ask a few reflective questions, then `extract_entry` → human confirmation via `interrupt()` → save. The `SqliteSaver` checkpointer uses thread_id = date, so you can resume today's conversation.
*Learn:* state, nodes, conditional edges, checkpointing, human-in-the-loop interrupts.

**M4 — Tools / ReAct agent. ✅** The `chat_agent` node becomes a tool-calling agent (`ToolNode` + `tools_condition`) with read-only `search_entries` (Chroma semantic search) and `get_entries_by_date`. Entries are embedded on save. llama3.1 vs qwen3 compared on tool calling with `evals/tool_probe.py`.
*Learn:* tool design (names, docstrings, arg schemas), agent loop, RAG, failure modes of small models calling tools.

**M5 — Summaries.** `diary summary --week|--month`: fetch range → map (condense each entry) → reduce (trends in mood, recurring themes, wins) → optionally save as a summary record.
*Learn:* map-reduce patterns, context-window budgeting on 8B models.

**M6 — Long-term memory.** A LangGraph `store` (namespace `("user", "profile")`) holds facts, goals and people. An `update_memory` node runs after each session to extract durable facts, and relevant memories are injected into the system prompt. A `remember_fact` tool handles explicit "remember that…" requests.
*Learn:* short-term (checkpointer) vs long-term (store) memory, memory write policies, avoiding memory pollution.

**M7 — Evals (the main learning payoff).** `evals/` harness, run with `uv run python evals/run_evals.py`:
1. **Component evals (deterministic):** entry extraction → field accuracy vs gold (mood within ±1, tags overlap F1, date correct).
2. **Retrieval evals:** seeded fake journal (~60 synthetic entries) + questions with known answer entry IDs → recall@k, MRR.
3. **Tool-use / trajectory evals:** `agentevals` trajectory match. Did the agent call `search_entries` for "what did I do last Tuesday?" and *not* call tools for small talk?
4. **Routing evals:** router picks correct mode on labeled utterances (accuracy + confusion matrix).
5. **LLM-as-judge (via `openevals`, local judge model):** empathy/non-judgmental tone, asks follow-up questions, no hallucinated past events (groundedness vs retrieved entries), safety (appropriate response to distress mentions).
6. **Multi-turn simulation:** a simulated-user LLM with a persona runs a full check-in; judge whether a complete, accurate entry was saved.
7. **Language-consistency eval:** formalize the M1 CJK-leak probe.
Results are saved as JSON with prompt version + model name, so runs can be compared (e.g. llama3.1:8b vs another model, prompt v1 vs v2). NOTES should cover eval pitfalls: judge bias, small datasets, and nondeterminism (run N times).

**M8 (optional polish).** Export entries to Markdown, `diary list`, safety guardrail node, local tracing viewer.

## Design principles (the "good agent" part)
- Deterministic code where possible; LLM only where judgment is needed (router can start rule-based + LLM fallback).
- Few, well-described tools with narrow arg schemas; validate tool inputs; return concise tool outputs.
- Human confirmation before writing an entry or a memory.
- Prompts versioned in `core/prompts.py` (`PROMPT_VERSION`); every eval run records versions.
- Graceful degradation when the 8B model fails structured output (retry with error feedback, then fallback).
- Privacy: everything in `./data/`, git-ignored.

## Verification
- `uv run pytest`: unit tests pass without Ollama.
- Manual: `uv run diary chat` → do a check-in, confirm the entry was saved (`diary list`). Restart the CLI and ask "what did I write today?" (checkpoint + search). Say "remember I'm training for a marathon", then ask about goals in a new session (store).
- `uv run diary summary --week` on seeded data.
- `uv run python evals/run_evals.py` produces a results JSON and a printed scorecard.

## Environment notes
- On Windows, if `uv` or `ollama` isn't found in a fresh shell, refresh PATH:
  `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`
