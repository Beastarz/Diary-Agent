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
| M3 LangGraph conversion | ⏭ next |
| M4–M8 | todo |

Change log:
- **M1:** default chat model switched from `qwen3:8b` to `llama3.1:8b`. qwen3 leaked Chinese tokens into 1–3 of every 9 English replies; llama3.1 leaked in 0 of 27 (see NOTES.md). Revisit qwen3 for tool calling in M4 using evals.
- **M2:** one entry per day (re-save asks before overwriting). Ollama structured output needs *all fields required* and *bounded lists* (see NOTES.md M2). Added deterministic `apply_guardrails()` for gratitude and mood words. `scripts/extract_probe.py` (7 cases) is the seed of the M7 extraction eval.

## Stack
- **LLM**: Ollama `llama3.1:8b` (default). `qwen3:8b` is also pulled, for comparisons and tool calling. Model name configurable via `.env` (`DIARY_CHAT_MODEL`).
- **Embeddings**: Ollama `nomic-embed-text` (768 dims).
- **Libraries**: `langchain-core`, `langchain-ollama`, `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-chroma` (local vector store), `pydantic`, `rich` + `typer` (CLI), `pytest`, `openevals` / `agentevals` (to add in M7).
- **Packaging**: `uv` with `pyproject.toml`; CLI entry point `diary = diary_agent.cli:app`.
- **Storage (all under `./data/`, git-ignored)**: `journal.db` (SQLite: entries table), `checkpoints.db` (LangGraph conversation state), `chroma/` (entry embeddings), `store.db` (long-term user memory).
- No cloud calls anywhere. LangSmith tracing is off by default; local tracing uses LangGraph's streaming plus a simple JSONL trace logger.

## Project layout (target)
```
diary_agent/
  pyproject.toml  .env.example  README.md  NOTES.md  PLAN.md
  scripts/smoke.py     # Ollama connectivity check            (exists)
  src/diary_agent/
    config.py          # model names, paths, env               (exists)
    llm.py             # get_chat_model(), get_embeddings()    (exists)
    prompts.py         # system prompts, versioned             (exists)
    chat.py            # M1 LCEL chain + ChatSession           (exists; superseded by graph in M3)
    cli.py             # `diary chat` (/save), `list`, `show`  (exists); `summary` in M5
    entry.py           # ExtractedEntry / JournalEntry schemas (exists)
    extract.py         # structured extraction + retries + guardrails (exists)
    db.py              # SQLite entries repo                   (exists)
    vectorstore.py     # Chroma index: add_entry(), search()
    memory.py          # long-term user-facts store (LangGraph BaseStore, SQLite-backed)
    tools.py           # @tool: save_entry, search_entries, get_entries_by_date, remember_fact
    graph/
      state.py         # JournalState (messages, mode, draft_entry, ...)
      nodes.py         # router, checkin, chat_agent, summarize, extract_entry, update_memory
      build.py         # StateGraph wiring + checkpointer + store
  evals/
    datasets/*.jsonl   # hand-written cases
    evaluators.py      # deterministic + trajectory + LLM-as-judge
    run_evals.py       # runs graph over datasets, writes evals/results/<ts>.json
  tests/               # plain unit tests using FakeListChatModel (no Ollama needed)
```

## Milestones
Each milestone ends runnable. Append a NOTES.md section (concepts, what to try, gotchas) and tick it off in README + this file.

**M0 — Setup. ✅** Ollama, uv, deps, `config.py`, `llm.py`, smoke script.
*Learn:* chat models, messages, streaming.

**M1 — Plain LangChain chat. ✅** `prompt | model` chain with a journaling-companion system prompt and in-memory history; CLI loop.
*Learn:* LCEL, prompt templates, why chat history must be managed explicitly.

**M2 — Structured entries.** `extract_entry` uses `model.with_structured_output(JournalEntry)` (pydantic: date, mood 1–10, mood_words, highlights, challenges, gratitude, tags, summary). Persist to SQLite via `db.py`.
*Learn:* structured output, schema design, validation/retry for small models.

**M3 — LangGraph conversion.** Rebuild as a `StateGraph`: `router → {checkin | chat_agent | summarize}`. The guided check-in is a small sub-flow: ask a few reflective questions, then `extract_entry` → human confirmation via `interrupt()` → save. The `SqliteSaver` checkpointer uses thread_id = date, so you can resume today's conversation.
*Learn:* state, nodes, conditional edges, checkpointing, human-in-the-loop interrupts.

**M4 — Tools / ReAct agent.** The `chat_agent` node becomes a tool-calling agent (`ToolNode` + `tools_condition`) with `search_entries` (Chroma semantic search + date filters), `get_entries_by_date` and `save_entry`. Entries are embedded on save. Compare llama3.1 against qwen3 on tool calling.
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
Results are saved as JSON with prompt version + model name, so runs can be compared (e.g. qwen3:8b vs llama3.1:8b, prompt v1 vs v2). NOTES should cover eval pitfalls: judge bias, small datasets, and nondeterminism (run N times).

**M8 (optional polish).** Export entries to Markdown, `diary list`, safety guardrail node, local tracing viewer.

## Design principles (the "good agent" part)
- Deterministic code where possible; LLM only where judgment is needed (router can start rule-based + LLM fallback).
- Few, well-described tools with narrow arg schemas; validate tool inputs; return concise tool outputs.
- Human confirmation before writing an entry or a memory.
- Prompts versioned in `prompts.py` (`PROMPT_VERSION`); every eval run records versions.
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
