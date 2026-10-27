# 3. LangGraph: state, nodes, edges, checkpoints and interrupts

How the chat loop became a graph that survives restarts, pauses for human approval, and handles multi-turn flows without writing an explicit loop.

**Where to look:** [`services/agent/state.py`](../../backend/src/diary_agent/services/agent/state.py), [`services/agent/nodes.py`](../../backend/src/diary_agent/services/agent/nodes.py), [`services/agent/graph.py`](../../backend/src/diary_agent/services/agent/graph.py) (the module docstring has the full ASCII graph), [`cli/app.py`](../../backend/src/diary_agent/cli/app.py) (`run_turn`, `answer_interrupt`)

---

### 1. A graph is nodes plus edges, and one run = one user turn
- **Concept:** A `StateGraph` node is a plain function `(state, runtime) -> partial update`. Edges say which node runs next. Each user message triggers one graph run from `START` to `END`.
- **In this repo:** `graph.py → build_graph()` wires `router`, `chat_agent`, `tools`, `checkin`, `extract`, `confirm`, `save`, `update_memory` and `confirm_memory`.
- **Why it matters:** Nodes return only the keys they change. That keeps each node small and easy to test.

### 2. State vs context: data you save vs tools you use
- **Concept:** **State** is the conversation's data. It's checkpointed after every node, so it must be serializable. **Context** holds dependencies (models, DB connections, today's date). It's passed fresh on every call and nodes read it as `runtime.context`.
- **In this repo:** `DiaryState` (messages, mode, `draft`, `pending_memories`, …) vs `DiaryContext` (chat model, extractor, SQLite connection, vector store, day). `draft` is stored as a `dict`, not an `ExtractedEntry`, because state has to serialize.
- **Rule of thumb:** *Could it be written to a database row and survive a restart?* Then it's state. *Is it something you call?* Then it's context. The long-term memory store is neither: it's passed to `compile(store=...)` and arrives as `runtime.store`.

### 3. Reducers decide how updates merge
- **Concept:** Each state key has a reducer. `messages` uses `add_messages`, which **appends**, and **replaces** a message that has the same `id`. Every other key is last-write-wins, unless you give it a custom reducer.
- **In this repo:** `confirm_memory` uses the replace-by-id rule to swap a placeholder `ToolMessage` for the real outcome. `state.py → merge_pending` is a custom appending reducer for `pending_memories`.
- **Why it matters:** Parallel tool calls can finish in the same step. A plain key accepts only one write per step, so two `remember_fact` calls would raise an error. The appending reducer makes it safe.

### 4. Two ways to route: conditional edges and `Command`
- **Concept:** Either a node writes a value and a conditional edge reads it, or a node returns `Command(goto=..., update=...)` to decide and update in one step.
- **In this repo:** `router` writes `route` and `add_conditional_edges` reads it. `confirm` and `confirm_memory` return `Command`, and their return type hint (`Command[Literal["chat_agent", "__end__"]]`) declares the possible targets.
- **Why it matters:** Edges keep the routing visible in `graph.py`. `Command` is handy when the decision depends on something only the node knows, like the human's answer.

### 5. Multi-turn flows without a loop
- **Concept:** Keep the flow's position in checkpointed state. Each new turn's router reads it and continues where the flow left off.
- **In this repo:** The check-in spans several user turns using `mode` and `checkin_step`. The *code* picks each topic from `CHECKIN_TOPICS`; the model only words the question.
- **Why it matters:** The flow is fully controlled by code and survives a restart halfway through. You can change the check-in by editing a list, without touching any prompt.
- **Trade-off:** Code control is rigid. The check-in always moves to the next topic, even right after the user shares hard news (found by simulated users, issue #15). And the model still sometimes drifts off the scripted topic (4 of 12 simulated check-ins skipped gratitude, #13).

### 6. The checkpointer is short-term memory
- **Concept:** A checkpointer saves state after every node, keyed by `thread_id`. Run the graph again with the same thread and you pick up where you left off.
- **In this repo:** `SqliteSaver` → `data/checkpoints.db`, with `thread_id` = today's date (`thread_config`). Restart `diary chat` and today's conversation is still there; tomorrow is a new thread.
- **Trade-offs:** One thread per day means one entry per day, which suits a diary. But a chat that runs past midnight stays on yesterday's thread until you restart.

### 7. `interrupt()` is human-in-the-loop, and the node runs twice
- **Concept:** Calling `interrupt(payload)` saves a checkpoint and stops the graph. The app shows the payload to the human and resumes with `Command(resume=value)`, which becomes `interrupt()`'s return value. **On resume, the node reruns from the top.**
- **In this repo:** `confirm` interrupts with the draft entry; the CLI asks "Save this entry?" and resumes with `True`/`False`. The write happens in a separate `save` node. `confirm_memory` uses the other valid pattern: it writes in the same node, but *after* `interrupt()`, which only the resumed run reaches.
- **Why it matters:** Any side effect *before* `interrupt()` happens twice. Put reads before it and writes after it, or in the next node. Bonus: Ctrl+C at "Save this entry?" leaves the thread paused, and the next `diary chat` asks again (`pending_interrupt`).

### 8. Keep non-conversation out of `messages`
- **Concept:** Anything in `messages` becomes part of the transcript that later steps read. Keep commands, status text and generated reports out of it.
- **In this repo:** `/save` and `/cancel` are handled by the router and never enter `messages`. Status lines go into a separate `notice` key. `/summary` points to the CLI instead of posting a review into the chat.
- **Why it matters:** A weekly review posted into chat would become part of today's transcript, and extraction would copy last week's events into today's entry. (This "contamination" theme comes up again in notes 4, 7 and 9.)

### 9. Streaming, and the bug only the real code path showed
- **Concept:** `graph.stream(..., stream_mode=["messages", "updates"])` gives you LLM tokens tagged with the node that produced them, plus each node's output.
- **In this repo:** `graph.py → stream_turn()` shows tokens only from `REPLY_NODES` (so extraction JSON isn't printed) and turns node updates into status lines ("Writing up your entry...", "Searching your journal...").
- **Why it matters (bug):** The first real run crashed: *"SQLite objects created in a thread can only be used in that same thread"*. When streaming, LangGraph runs sync nodes in worker threads. The unit tests used `invoke` and passed. Fix: `check_same_thread=False`, plus a regression test through `stream_turn`. **Test the same code path your users run.**

---

### Try it
- Start `/checkin`, quit halfway, restart `diary chat`: it continues at the same question.
- Press Ctrl+C at "Save this entry?", then run `diary chat` again.
- Print the graph as Mermaid and paste it into mermaid.live:
  `uv run python -c "from diary_agent.services.agent.graph import build_graph; print(build_graph().get_graph().draw_mermaid())"`
- Inspect every step of today's thread with `graph.get_state_history(config)`.

**Full story:** [NOTES.md](../NOTES.md) → M3, and M6 for the memory nodes.
