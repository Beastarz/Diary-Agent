# Learning notes: building an AI agent, one lesson at a time

This repo is a **local diary agent**: you chat with it about your day, it writes a structured journal entry, searches past entries, writes weekly/monthly reviews, and remembers lasting facts about you. It runs entirely on a free local model (Ollama `llama3.1:8b` on an 8 GB GPU) and was built milestone by milestone (M0–M7) to learn **LangChain, LangGraph, tool use, RAG, memory and evals**.

These notes pull the lessons out of the code and reorganise them **by topic** rather than by milestone, so you can learn one subject end to end. Almost every point comes from a real bug or an eval number from building this project, not just from the docs.

## Reading order

| # | Note | What you'll learn | Milestones |
|---|---|---|---|
| 1 | [LLM fundamentals](01-llm-fundamentals.md) | Chat models, messages, LCEL, why models are stateless, embeddings, choosing a model | M0–M1 |
| 2 | [Structured output & guardrails](02-structured-output.md) | Getting reliable JSON out of an 8B model, schema design, retries, code guardrails | M2 (+M5, M6) |
| 3 | [LangGraph](03-langgraph.md) | State, nodes, edges, reducers, checkpoints, human-in-the-loop `interrupt()`, streaming | M3 (+M6) |
| 4 | [Tools & the ReAct loop](04-tools-and-agents.md) | Tool design, the agent loop, code gates, loop budgets, tools that never write | M4 (+M6, M7) |
| 5 | [RAG & embeddings](05-rag-and-embeddings.md) | Vector search, what to embed, relevance cutoffs, similarity ≠ sameness | M4 (+M6, M7) |
| 6 | [Context budgets & summaries](06-context-and-summaries.md) | Context windows, map-reduce, when *not* to add an LLM step | M5 |
| 7 | [Long-term memory](07-memory.md) | Short- vs long-term memory, write policies, avoiding memory pollution | M6 |
| 8 | [Evals](08-evals.md) | Tests vs evals, the eval loop, labels, checking the checker, noise | M1–M7 |
| 9 | [LLM judges & simulated users](09-llm-judges-and-simulation.md) | Rubrics, calibrating a judge against hand grades, whole-conversation tests and their pitfalls | M7 |
| 10 | [Agent design principles](10-agent-design-principles.md) | The patterns that kept coming back, with the evidence for each | all |

New to LLM apps? Read 1 → 10 in order. Already know LangChain? Start at 3, then read 8, 9 and 10.

## How each note is laid out

Every point has three parts:
- **Concept:** the idea in plain words.
- **In this repo:** the file and symbol where you can see it.
- **Why it matters:** the lesson, usually a bug we hit or a number an eval produced.

Each note ends with **Try it** exercises you can run against the code.

## The agent at a glance

```
diary chat ──► LangGraph StateGraph (one run per user message)
   START → router (rule-based)
     ├─ chat_agent ⇄ tools        ReAct loop: search_entries, get_entries_by_date, remember_fact
     ├─ checkin                   guided questions; the code picks the topic, the model words it
     └─ extract → confirm ──(you say yes)──► save → update_memory → confirm_memory
                   interrupt()                SQLite + Chroma   LLM      interrupt(), then the store

diary summary --week|--month ──► day lines (code) → [collapse] → reduce (LLM) → you confirm → save
```

Storage (all in `data/`, git-ignored): `journal.db` (entries + summaries), `checkpoints.db` (today's conversation), `chroma/` (entry vectors), `store.db` (long-term memory).

## Conventions

- Code links point into [`backend/src/diary_agent/`](../../backend/src/diary_agent/). The layers are `cli/` → `services/` → `repositories/` → `models/`, and `core/` (config, LLM factories, prompts) is used everywhere.
- [`docs/NOTES.md`](../NOTES.md) is the original per-milestone narrative with the full eval tables. These notes link to it when you want the whole story.
- [`docs/PLAN.md`](../PLAN.md) is the roadmap and status.
- Numbers like "75% → 0%" come from small eval sets (often 10–25 cases × 2–3 runs). Read them as *direction*, not precision.
