# Learning notes

## M0: Chat models, messages, streaming
- **Where to look:** `backend/src/diary_agent/core/llm.py`, `scripts/smoke.py`
- A LangChain **chat model** (`ChatOllama`) takes a list of **messages** (`SystemMessage`, `HumanMessage`, `AIMessage`) and returns an `AIMessage`.
- `.invoke()` returns the whole reply. `.stream()` yields chunks as they are generated.
- Keep model creation in one factory (`get_chat_model`). The rest of the app depends only on the `BaseChatModel` interface, so switching models or providers is a config change.
- qwen3 is a *reasoning* model. `reasoning=False` turns off its hidden "thinking", which is faster and fine for small talk. Try `DIARY_REASONING=true` later for summaries.
- **Embeddings** turn text into vectors (768 dims for nomic-embed-text). M4 uses them for semantic search.

## M1: LCEL and chat history
- **Where to look:** `services/chat.py`, `cli/app.py`, `backend/tests/services/test_chat.py`
- `prompt | model | parser` is **LCEL**. Each piece is a `Runnable`, and `|` chains them. Every Runnable gets `invoke / stream / batch` for free.
- `ChatPromptTemplate` fills `{today}` and `{input}`. `MessagesPlaceholder("history")` slots in the prior messages.
- **The model is stateless.** It remembers nothing between calls, so "memory" just means resending the history each time. Our `ChatSession` keeps it in a Python list, and it is lost when the process exits. M3 fixes that with a LangGraph checkpointer.
- **Testing without an LLM:** `FakeListChatModel` returns canned replies, so the plumbing can be unit-tested quickly and deterministically. Keep these tests separate from *evals*, which measure how good the real model's behavior is.

### Observed failure: first eval idea
In the first real run, qwen3 answered in English but slipped in a Chinese word: "What made that time with your sister特别 special". Small multilingual models do this.

1. **Tried a prompt fix** (v1.1: "reply in the user's language"). The next run *still* leaked. **A prompt change is only a guess until you measure it.**
2. **Measured it.** A small probe sent 3 journal messages × 3 repeats to each config and used a regex to flag any Chinese/Japanese/Korean (CJK) characters:

   | model / settings | replies with CJK |
   |---|---|
   | qwen3:8b, temp 0.7, no reasoning | 2/9 |
   | qwen3:8b, temp 0.3 | 3/9 |
   | qwen3:8b, reasoning on | 1/9 |
   | qwen3:8b, Qwen-recommended sampling | 3/9 |
   | llama3.1:8b (3 settings) | **0/27** |

3. **Decision:** default chat model is now `llama3.1:8b`. Sampling settings didn't fix qwen3. The model was the problem.

This was a tiny **eval**: fixed inputs, a deterministic checker, several repeats (outputs are random), and a table comparing configurations. M7 grows it into a proper harness. Open question for M4: qwen3 is usually better at tool calling. Would per-task models (llama for chat, qwen for tools) be worth the complexity? We'll let evals decide.

### Try this
- Change the system prompt tone and compare replies.
- Set `DIARY_CHAT_MODEL=qwen3:8b` and compare.
- Chat for 30+ turns. The history grows without limit. What happens to speed? (Hint: context window. M3/M6 handle trimming.)

## M2: Structured output, schema design, guardrails
- **Where to look:** `models/entry.py` (schemas), `services/extraction.py` (extraction + retries + guardrails), `repositories/entry_repository.py`, `cli/app.py` (`/save`, `list`, `show`), `evals/extract_probe.py` (mini eval)
- `model.with_structured_output(Schema, method="json_schema", include_raw=True)` returns a Runnable that outputs `{"raw", "parsed", "parsing_error"}`. `include_raw` lets us see *why* parsing failed and feed that back to the model for a retry.
- **Two schemas:** `ExtractedEntry` holds what the LLM must infer. `JournalEntry` adds what code already knows: date, transcript, model, prompt version. Never ask a model for something you can compute.
- **Human confirmation before writing:** `/save` shows the entry and asks y/n. Saving the same day again asks before overwriting.

### How Ollama structured output actually works (and 3 bugs it caused)
Ollama compiles the JSON schema into a **grammar** that constrains which tokens the model may pick. It guarantees valid *structure*, not correct *content*. Each of the three bugs below was found with the probe script, not by reading code:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | mood `null` in 4/6 cases; tags and mood_words always empty | The grammar only forces **required** fields. Fields with defaults were optional, so the model skipped them. Raw output was just `{"summary":…, "tags":…}` | Make every field required. Express "nothing" as `[]` or `null` |
| 2 | Probe hung for >10 min at 98% GPU | An unbounded list let the model repeat tags forever (`"surgery", "went", "well", "surgery", …`) | `max_length` on every list + `num_predict=512` safety cap |
| 3 | Field descriptions seemed to have no effect | The grammar doesn't show descriptions to the model; they only shape structure | `field_guide()` puts the schema descriptions into the prompt |

Also: **field order matters.** The model writes JSON top to bottom, so evidence (highlights, challenges) comes first and the judgement (mood) comes after.

### The eval loop in practice
Scores from `extract_probe.py` (7 cases × 3 runs, llama3.1:8b):

| version | mood | no_invented | mentions | what changed |
|---|---|---|---|---|
| v1.2 | 33% | 83% | 100% | baseline (optional fields skipped) |
| v1.3 | 67% | 83% | 83% | required fields, list caps, field guide in prompt |
| v1.3 + eval fix + gratitude guardrail | 100% | 100% | 100% | see below |
| + new case `assistant_embellishes` | 100% | 86% | 100% | a real CLI run exposed a new failure |
| v1.4 prompt tweak | 100% | **81%** | 86% | fixed target 0→2/3 but **regressed** `no_mood_stated` |
| v1.4 + mood-word guardrail | 100% | 100% | 86% | current |

Lessons:
1. **Sometimes the eval is wrong, not the model.** `no_mood_stated` expected `null`, but our own schema says "null only if no events". Mood labels also changed from exact scores to *ranges*, because demanding "6" when 5–7 is equally defensible measures noise. Change labels only with a written reason (see comments in the probe). Otherwise you're moving the goalposts.
2. **Prompt fixes regress other cases.** v1.4 fixed one case and broke another. That's why you rerun the *whole* suite after every change, never just the failing case.
3. **Prefer deterministic guardrails for rules you can check in code.** The model kept inventing gratitude and borrowing the assistant's words as the user's feelings. `apply_guardrails()` enforces both rules with plain string checks: 100% reliable, unit-tested, and impossible to regress through prompt changes. The trade-off is that it drops some legitimate inferences (e.g. "happy" when the user only said "great day"). Here that's fine, because the spec says "own words".
4. **Real usage finds failures your test cases don't.** `assistant_embellishes` came from a CLI run where "invigorating" (the assistant's word) became the user's mood. Every bug seen in the wild becomes an eval case.
5. **Temperature 0 isn't fully deterministic.** `good_day_explicit` flips between "Took partner out for sushi" and "celebrated with partner" with an identical prompt, because GPU math isn't bit-exact. Always do several runs.
6. **100% means the eval is too easy, not that the model is perfect.** The probe still misses things: "8 out of 10" listed as a *highlight*, and a `work` tag on a dentist trip. M7 needs harder cases and checks on tags and highlights.

### Try this
- `uv run python evals/extract_probe.py --model qwen3:8b`: how does it compare?
- Remove `max_length` from `tags` and rerun `gratitude_only` (be ready to Ctrl+C).
- Write a case you think will fail, e.g. sarcasm: "Oh *great*, another Monday." Does the mood come out right?
- `uv run diary show --transcript` to see what was stored alongside the entry.

## M3: LangGraph: state, nodes, edges, checkpoints, interrupts
- **Where to look:** `services/agent/state.py` (state + context), `services/agent/nodes.py`, `services/agent/graph.py` (wiring, checkpointer, streaming), `cli/app.py` (`run_turn`, `ask_to_save`), `tests/services/agent/`
- **The graph replaces `ChatSession`.** `START -> router -> {chat_agent | checkin | summarize | extract -> confirm -> save} -> END`. One `invoke` = one user turn.
- **State vs context.** `DiaryState` is the conversation's *data* (messages, mode, draft). It is checkpointed after every node, so it must be serializable; that's why `draft` is a dict, not an `ExtractedEntry`. `DiaryContext` holds the *dependencies* (models, DB connection, date). It is passed fresh on each call via `context=` and nodes read it as `runtime.context`. Rule of thumb: if it should survive a restart, it's state; if it's a tool you use, it's context.
- **Reducers.** `messages` uses `add_messages`, so a node returning `{"messages": [msg]}` *appends*. Every other key is overwritten. Nodes return only the keys they change.
- **Conditional edges vs `Command`.** The router writes `route` and a conditional edge reads it: the decision is data you can see in the state. `confirm` returns `Command(goto=...)` instead, deciding and updating in one step. Both are fine; edges keep routing visible in `graph.py`.
- **Multi-turn flows without loops.** The check-in spans 5 user turns, but the graph never loops. `mode` and `checkin_step` sit in the checkpointed state, so each new turn's router knows where we are. The code picks the next topic (`CHECKIN_TOPICS`); the model only words the question.
- **Commands never enter `messages`.** `/save` is handled by the router and doesn't pollute the transcript the extractor reads. The CLI shows status through a separate `notice` key.
- **Checkpointer = short-term memory.** `SqliteSaver` writes `data/checkpoints.db`, and `thread_id` = today's date. Restart `diary chat` and the conversation is still there. Tomorrow is a new thread. Try `graph.get_state_history(config)` to see every step.
- **`interrupt()` = human in the loop.** `confirm` calls `interrupt(payload)`: the graph checkpoints and stops. The CLI shows the draft and resumes with `Command(resume=True/False)`, which becomes `interrupt()`'s return value. **On resume the node reruns from the top**, so code before `interrupt()` must be safe to repeat. The DB *read* is there; the *write* is in a separate `save` node. Ctrl+C at "Save this entry?" leaves the thread paused, and the next `diary chat` asks again.
- **Streaming.** `stream_mode=["messages", "updates"]`: "messages" gives LLM tokens tagged with `langgraph_node` (so extraction JSON isn't printed), and "updates" gives each node's output (used for the "Writing up your entry..." status).

### Bug found by running, not by tests
The first real run crashed at `confirm`: `SQLite objects created in a thread can only be used in that same thread`. **LangGraph runs sync nodes in worker threads** when streaming, and `sqlite3` connections refuse cross-thread use by default. The unit tests used `invoke` and passed anyway. Fix: `check_same_thread=False` (safe, because one thread's steps run one node at a time), plus a regression test through `stream_turn`, the CLI's code path. It was checked to fail without the fix. **Lesson: test the same code path users run.**

### Observed model failure (for M7)
After a restart, "what was the best part of my day?" got: "...the laughter and conversation you had with your friend". The user never mentioned laughter. That is a **hallucinated past detail**, exactly what the groundedness judge in M7 should catch. It's also why M4 gives the agent a `search_entries` tool instead of letting it improvise.

### Known limits
- The keyword router misfires on "I had to check in at the hotel". M7 routing evals will measure this before an LLM fallback is added.
- The whole day's history is resent every turn. Fine for one day; trimming comes with M6.
- A conversation that runs past midnight stays on the old thread until you restart.

### Try this
- Start a `/checkin`, quit halfway, restart: it continues at the same question.
- At "Save this entry?" press Ctrl+C, then run `diary chat` again.
- `uv run python -c "from diary_agent.services.agent.graph import build_graph; print(build_graph().get_graph().draw_mermaid())"`, then paste the output into mermaid.live.
- Change `CHECKIN_TOPICS` and see the check-in change, with no prompt edit.

## M4: Tools, the ReAct loop, RAG, and measuring tool use
- **Where to look:** `services/agent/tools.py`, `repositories/vector_repository.py`, `services/agent/nodes.py` (`chat_agent`, `calendar_hint`, `LOOKUP_REQUEST`), `services/agent/graph.py`, `evals/tool_probe.py`, `tests/services/agent/test_tools.py`
- **The loop.** `chat_agent <-> tools`. The model replies; if the reply has `tool_calls`, `tools_condition` sends it to `ToolNode`, which runs the tools and appends `ToolMessage`s; then the model is called again with the results. With no tool calls, the turn ends. That's ReAct: reason → act → observe → repeat. It's still one `invoke` per user turn, just with more steps inside.
- **A tool = name + docstring + arg schema.** That's *all* the model sees (print `convert_to_openai_tool(tool)` to check). The docstring says *when* to use the tool, not just what it does. `ToolRuntime` gives tools the same `DiaryContext` as nodes and is hidden from the schema.
- **Tool errors are messages.** Bad input returns `"Error: ... Use YYYY-MM-DD."` so the model can retry. `ToolNode` also turns uncaught exceptions into `ToolMessage`s by default.
- **Read-only tools on purpose.** There's no `save_entry` tool: writes still go through extract → `interrupt()` → save, so the model can never write without a human yes.
- **RAG = retrieve, then generate.** Entries are embedded on save (summary + highlights + challenges + tags, not the noisy transcript) into Chroma with `id = date`, so re-saving replaces the vector. SQLite stays the source of truth: search returns dates, and the tool reads the full entries from SQLite. `diary reindex` rebuilds the index. nomic-embed-text wants `search_document:` / `search_query:` prefixes.
- **Tool traffic stays out of today's entry.** Tool results are *past* entries. `format_transcript` drops tool messages, or the extractor would copy old days into today. There's a test for it.
- **Guard the loop.** `MAX_TOOL_ROUNDS = 3`. After that the model is called without tools and has to answer.

### What the eval found (evals/tool_probe.py)
14 cases (date questions, topic questions, small talk) × 3 runs, one real graph turn each, against a seeded 10-entry journal with "today" fixed. Every fix below came from a number, not a hunch:

| Change | Failure it fixed (llama3.1) | Before → after |
|---|---|---|
| Date cheat-sheet in the prompt (`calendar_hint`) | "Monday this week" → Oct 17. 8B models can't do calendar math | date args 50% → 90% |
| Removed optional date args from `search_entries` | `start_date: "0000-00-00"`, then "I need a valid start date" | fewer args, fewer ways to fail |
| Distance cutoff (`MAX_DISTANCE = 0.43`) | "When did I go skiing?" → an invented Niseko trip. Nearest-neighbour search always returns k results | admits "nothing found" 3/3 |
| Code gate: tools offered only for questions/requests (`LOOKUP_REQUEST`) | Tool calls on "thanks" and "I'm anxious". **A prompt rule changed nothing (75% → 75%)** | false tool calls 75% → 0% |

Final llama3.1: tool choice 93%, args 83%, grounded answers 90%, 0% false tool calls, ~1.7 s/turn.

**llama3.1 vs qwen3 (same prompt v1.7, before the gate):** qwen3 never called tools on small talk, but missed 40% of needed calls (llama 7%), got date args right 43% of the time (llama 90%), and invented details without looking ("a small bowl and a decorative tile"). It even misread the cheat-sheet ("Tuesday, October 19"). **llama3.1 stays the default.** The M1 CJK-leak check was clean for both on these short replies.

### Lessons
- **When a prompt rule doesn't hold, move the decision into code.** The model is good at *which* tool; the code is better at *whether* tools are on offer at all, and at dates.
- **Two models, opposite biases.** llama over-calls, qwen3 under-calls. A single "accuracy" number hides that, so split it into missed-call and false-call rates.
- **Separate retrieval from reasoning.** The eval's model-free retrieval check had recall 4/4, so the remaining failures are the model's, not Chroma's.
- **Check your checker.** Several "failures" were the keyword checker missing a correct answer ("that dinner with your sister on Wednesday"). Strict labels can also be debatable: "running records *recently*" via a date range got the right answer, but was scored as the wrong tool. That's M7's job (LLM judge, more cases).
- **n is tiny.** 14 cases × 3 runs, and the threshold was calibrated on 10 entries. Treat the numbers as directional. Results JSON records the model, prompt version and git commit.

### Known limits
- `LOOKUP_REQUEST` needs a "?" or a request verb at the start: "when did I last see my sister" without a "?" gets no tools.
- `MAX_DISTANCE` is calibrated on 10 synthetic entries; an extra word in the query ("pottery class *today*") can push a real match past it.
- The assistant's final reply about a past day stays in today's transcript, so extraction could still pick up past events from it. Watch for this in the M7 extraction eval.

### Try this
- `uv run python evals/tool_probe.py --runs 3` (both models, ~4 min). Then delete the `LOOKUP_REQUEST` check in `chat_agent` and run `--models llama3.1:8b` again.
- Set `MAX_DISTANCE = 1.0` and ask about skiing.
- In `diary chat`, ask "what did I write yesterday?" and watch the "Searching your journal..." status.
