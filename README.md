# diary-agent

A local-first, chat-based daily journal agent. Built step by step to learn LangChain, LangGraph, agent design, and agent evals. Everything runs on your machine through [Ollama](https://ollama.com); no data leaves it.

## Setup
```powershell
winget install Ollama.Ollama; winget install astral-sh.uv
ollama pull llama3.1:8b; ollama pull nomic-embed-text
ollama pull qwen3:8b     # optional, for model comparisons
uv sync
copy backend\.env.example backend\.env   # optional: change model etc.
```

All commands below run from the repo root. It's a uv workspace, so there's one `.venv` and one `uv.lock` for everything.

## Structure
```
backend/                 Python app (package: diary_agent)
  src/diary_agent/
    core/                config, LLM factories, prompts
    models/              pydantic data models (ExtractedEntry, JournalEntry)
    repositories/        data access: SQLite entries + Chroma search index
    services/            business logic: LangGraph agent (services/agent/), entry extraction
    cli/                 Typer CLI (`diary` command)
    api/                 HTTP API for the frontend (empty for now)
  tests/                 unit tests, mirroring src/ (no Ollama needed)
frontend/                future web UI (placeholder)
evals/                   model-quality evals (need Ollama)
scripts/                 dev utilities
docs/                    PLAN.md (roadmap), NOTES.md (learning notes)
```
Dependencies point one way: `cli/api → services → repositories → models`, and everything can use `core`.

## Use
```powershell
uv run diary chat                  # talk to your journal; resumes today's conversation on restart
                                   #   /checkin guided questions, /save write today's entry, /cancel stop a check-in
uv run diary list                  # recent entries
uv run diary show [YYYY-MM-DD]     # one entry (--transcript to include the chat)
uv run diary reindex               # rebuild the search index from saved entries
uv run python evals/extract_probe.py   # mini eval of entry extraction (needs Ollama)
uv run python evals/tool_probe.py      # tool-use eval, llama3.1 vs qwen3 (needs Ollama)
uv run pytest                      # unit tests (no Ollama needed)
uv run python scripts/smoke.py     # check Ollama connectivity
```

## Roadmap
- [x] M0 Setup: Ollama, uv, model factories
- [x] M1 Plain LangChain chat (LCEL, in-memory history)
- [x] M2 Structured journal entries + SQLite
- [x] M3 LangGraph: state graph, checkpointing, human-in-the-loop
- [x] M4 Tools + RAG over past entries (ask "what did I do yesterday?" in chat)
- [ ] M5 Weekly/monthly summaries
- [ ] M6 Long-term memory about you
- [ ] M7 Evals: component, retrieval, trajectory, LLM-as-judge, simulated users

See [`docs/PLAN.md`](docs/PLAN.md) for the full design and [`docs/NOTES.md`](docs/NOTES.md) for what each milestone teaches.
