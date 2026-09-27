# diary-agent

A local-first, chat-based daily journal agent. Built step by step to learn LangChain, LangGraph, agent design, and agent evals. Everything runs on your machine through [Ollama](https://ollama.com); no data leaves it.

## Setup
```powershell
winget install Ollama.Ollama; winget install astral-sh.uv
ollama pull llama3.1:8b; ollama pull nomic-embed-text
ollama pull qwen3:8b     # optional, for model comparisons
uv sync
copy .env.example .env   # optional: change model etc.
```

## Use
```powershell
uv run diary chat                  # talk to your journal; type /save to write today's entry
uv run diary list                  # recent entries
uv run diary show [YYYY-MM-DD]     # one entry (--transcript to include the chat)
uv run python scripts/extract_probe.py   # mini eval of entry extraction (needs Ollama)
uv run pytest                      # unit tests (no Ollama needed)
uv run python scripts/smoke.py     # check Ollama connectivity
```

## Roadmap
- [x] M0 Setup: Ollama, uv, model factories
- [x] M1 Plain LangChain chat (LCEL, in-memory history)
- [x] M2 Structured journal entries + SQLite
- [ ] M3 LangGraph: state graph, checkpointing, human-in-the-loop
- [ ] M4 Tools + RAG over past entries
- [ ] M5 Weekly/monthly summaries
- [ ] M6 Long-term memory about you
- [ ] M7 Evals: component, retrieval, trajectory, LLM-as-judge, simulated users

See `PLAN.md` for the full design and `NOTES.md` for what each milestone teaches.
