# 4. Tools and the ReAct loop

How the agent decides to look things up, how to design tools a small model can use, and how to keep a tool-calling model from going off the rails.

**Where to look:** [`services/agent/tools.py`](../../backend/src/diary_agent/services/agent/tools.py), [`services/agent/nodes.py`](../../backend/src/diary_agent/services/agent/nodes.py) (`chat_agent`, `tools_for`, `calendar_hint`, `confirm_memory`), [`services/agent/graph.py`](../../backend/src/diary_agent/services/agent/graph.py), [`evals/tool_probe.py`](../../evals/tool_probe.py)

---

### 1. The ReAct loop is a cycle in the graph
- **Concept:** ReAct = reason → act → observe → repeat. The model replies; if the reply contains `tool_calls`, the tools run, their results are appended as `ToolMessage`s, and the model is called again with the results.
- **In this repo:** `chat_agent` → `tools_condition` → `ToolNode` → `route_after_tools` → back to `chat_agent`. With no tool calls, the turn ends.
- **Why it matters:** An "agent" is just a conditional edge plus an edge back. It's still one graph run per user message, just with more steps inside.

### 2. A tool is a name, a docstring and an argument schema
- **Concept:** That's *everything* the model sees. The docstring should say **when** to use the tool, not just what it does.
- **In this repo:** `@tool` functions with `Annotated[...]` argument descriptions: `search_entries(query)`, `get_entries_by_date(start, end)`, `remember_fact(content, kind)`. `remember_fact`'s docstring ends with "Use this ONLY when the user explicitly asks you to remember something."
- **Why it matters:** Print `convert_to_openai_tool(tool)` to see exactly what the model gets. If it's not in that output, the model doesn't know about it.

### 3. Hidden, injected arguments
- **Concept:** Some arguments are for your code, not the model. LangGraph injects them at call time and leaves them out of the schema.
- **In this repo:** `runtime: ToolRuntime[DiaryContext]` gives tools the same context as nodes (DB, vector store, day), plus `runtime.state`, `runtime.store` and `runtime.tool_call_id`.
- **Why it matters:** The model can't fill in a database connection, and shouldn't be able to pick which user's data to read.

### 4. Keep arguments narrow, validate in code, return errors as text
- **Concept:** Fewer arguments mean fewer ways to fail. Validate inputs in the tool and return an error *string* the model can act on, rather than raising.
- **In this repo:** `get_entries_by_date` returns `"Error: ... Use YYYY-MM-DD."`. `remember_fact` rejects one-word content or content starting with "I". `ToolNode` also turns uncaught exceptions into `ToolMessage`s.
- **Why it matters (bugs):** `search_entries` once had optional date arguments, and the model filled them with `0000-00-00` and then complained it needed a valid date. Removing them fixed it. For "Remember that I'm vegetarian", llama sent `content='I'`; the error message makes it call the tool again correctly.

### 5. Code chooses the menu, the model chooses from it
- **Concept:** Small models are decent at picking *which* tool, but bad at deciding *whether* any tool is needed. Let code decide which tools are on offer for each turn.
- **In this repo:** `nodes.py → tools_for(text)`. `LOOKUP_REQUEST` (a "?" or a leading verb like "tell", "show", "find") unlocks the read tools. `MEMORY_REQUEST` (an imperative "remember that…" / "don't forget…", but not "remember when…") unlocks `remember_fact`. Anything else gets a plain call with no tools bound at all.
- **Why it matters (evals):** llama called tools on "thanks" and "I'm anxious". A prompt rule against it changed nothing (**75% → 75%**); the code gate got it to **0%**. Offering `remember_fact` on any "remember" saved "I remember the beach trip" as a memory 4/4 times; the per-tool gate made it 0/4.

### 6. Do the hard deterministic work for the model
- **Concept:** Don't ask an 8B model to do date arithmetic. Compute it and put the answer in the prompt.
- **In this repo:** `nodes.py → calendar_hint(day)` puts a cheat-sheet into `TOOLS_SYSTEM`: today, yesterday, this week's and last week's date ranges, this month, and the last two weeks of days with their weekday names. The model only has to *copy* a date, not compute one.
- **Why it matters:** "Monday this week" came out as the wrong date. With the cheat-sheet, correct date arguments went from **50% → 90%**.

### 7. Give the loop a budget
- **Concept:** A model can keep calling tools forever. Cap the rounds, then force an answer.
- **In this repo:** `MAX_TOOL_ROUNDS = 3`, counted by `_tool_rounds_this_turn`. After that, `chat_agent` calls the model *without* tools and strips any leftover `tool_calls` (otherwise the graph would route back to `ToolNode`). `GAVE_UP` is the fallback message.
- **Why it matters:** Every loop in an agent needs an exit that doesn't depend on the model behaving.

### 8. Tools don't write: they propose, and a human approves
- **Concept:** Never give a model a tool that writes data without a human check. A tool can *stage* a change in state; a separate node asks the human and then writes.
- **In this repo:** There is no `save_entry` tool; entries go extract → `interrupt()` → save. `remember_fact` returns a `Command` that adds a proposal to `pending_memories` plus a placeholder `ToolMessage`. `confirm_memory` asks "Remember this?", writes only what was approved, and **replaces the placeholder by id** with the real outcome, so the model's reply ("Got it") reflects what actually happened.
- **Why not call `interrupt()` inside the tool?** `ToolNode` runs parallel tool calls in a thread pool, and several interrupts racing inside one node is fragile. Staging in state is simple and safe.

### 9. Tool traffic must not leak into downstream steps
- **Concept:** Tool results bring *other* data into the conversation (here, past diary entries). Anything that later reads the conversation must filter it out.
- **In this repo:** `services/extraction.py → format_transcript()` drops tool messages, tool-call messages, **and the assistant's answer after a tool result** (until the user speaks again).
- **Why it matters (M7 eval):** The assistant said "Last Saturday you ran a 10k", and extraction put "ran a 10k" into *today's* highlights every time. A "today only" prompt rule fixed that case but broke two others. The code fix works because the code *knows* which replies came from a lookup; a prompt can only ask the model to guess.

---

### Try it
- `uv run python evals/tool_probe.py --runs 3`. Then delete the `LOOKUP_REQUEST` check in `tools_for` and run it again: watch false tool calls come back.
- `uv run python evals/memory_probe.py --part tool`, then change `MEMORY_REQUEST` to `\bremember\b` and rerun.
- In `diary chat`, ask "what did I write yesterday?" and watch the "Searching your journal..." status.

**Full story:** [NOTES.md](../NOTES.md) → M4 ("What the eval found"), M6 (`remember_fact`), M7 ("First fix through the loop").
