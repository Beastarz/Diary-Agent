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
- Set `DIARY_CHAT_MODEL` to another Ollama model and compare.
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
- `DIARY_CHAT_MODEL=<other model> uv run python evals/run_evals.py --only extraction`: how does it compare? (`extract_probe.py` was folded into the M7 harness.)
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

**llama3.1 vs qwen3 (same prompt v1.7, before the gate):** qwen3 never called tools on small talk, but missed 40% of needed calls (llama 7%), got date args right 43% of the time (llama 90%), and invented details without looking ("a small bowl and a decorative tile"). It even misread the cheat-sheet ("Tuesday, October 19"). **llama3.1 stays the default, and qwen3 was dropped from the project after this.** The M1 CJK-leak check was clean for both on these short replies.

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
- `uv run python evals/run_evals.py --only tools --runs 3` (the M7 harness now holds these cases; `DIARY_CHAT_MODEL=<model>` to compare). Then delete the `LOOKUP_REQUEST` check in `chat_agent` and run `--models llama3.1:8b` again.
- Set `MAX_DISTANCE = 1.0` and ask about skiing.
- In `diary chat`, ask "what did I write yesterday?" and watch the "Searching your journal..." status.

## M5: Map-reduce summaries and context budgeting
- **Where to look:** `services/summary.py`, `models/summary.py`, `repositories/summary_repository.py`, `cli/app.py` (`summary`), `evals/summary_probe.py`, `tests/services/test_summary.py`
- **The shape.** *map* condenses each entry into a ~2-sentence digest (one LLM call per entry, via `writer.batch()`, which is LCEL's map). *collapse* merges groups of digests, but only when they're still over budget. *reduce* writes the review as structured output (`PeriodSummary`). Each call has a known, bounded input size, which is the whole point.
- **Why not one big prompt?** Ollama's default context window is a few thousand tokens, and anything past it is **cut silently**: the model just never sees the start of the month. So every summary call sets `num_ctx=8192`, and they all use the same value, because changing `num_ctx` between requests makes Ollama reload the model.
- **The budget, in tokens** (estimated as chars / 4, which is rough but enough for a budget): `MAP_INPUT_TOKENS = 2000` per entry, `REDUCE_INPUT_TOKENS = 3000` of digests. Cheap cuts come first: the map input keeps only the `User:` lines of the transcript, since the assistant's words aren't facts about the day and are about half the tokens.
- **Code does the numbers.** Mood average, lowest/highest day and tags that recur on 2+ days are computed and passed to the reduce prompt as facts. The date and mood prefix on each digest also comes from code, so the reduce step can't lose them.
- **Summaries stay out of the chat.** `/summary` points to the CLI. A review posted into today's conversation would become part of today's transcript, and extraction would copy last week's events into today's entry.
- **Saving needs a yes**, like entries. It's one row per `(period, start_date)` in `summaries`, and `--saved` shows it again without an LLM call.

### What the probe found (evals/summary_probe.py)
3 cases (a full week, a 10-entry month, and a sparse week with 2 entries) × 3 runs, through the real `summarize_period`, on tool_probe's seeded October journal. Each fix came from a number:

| Change | Failure it fixed (llama3.1) | Before → after |
|---|---|---|
| Condense prompt: "at most 3 sentences, one is fine", "keep the user's wording" | "2-3 sentences" made it pad a short entry: "*Your first project is now complete*" became "completed pottery class" in the review | embellish → 9/9 |
| **Removed** a computed "Mood trend: down (6.8 → 6.0)" line | With it, the review named both the low and the high day in 2/6 runs; without it, 6/6. The model narrated the direction and dropped the specifics | mood 2/6 → 6/6 |
| Code writes `mood_trend` when fewer than 3 days are rated | "remained steady at 7/10 throughout the period" for 2 days out of 7. **A prompt rule changed nothing (0/3)** | trend overclaim gone |
| Guardrail drops wins/challenges containing "mood" | "good mood" listed as a win (3/3 → 1/3 after a field-description fix, then 0 with the guardrail) | grounded 8/9 → 9/9 |

Final: grounded 9/9, low+high days named 6/6, recurring themes 3/3, no embellishment 9/9, sparse-week wording 2/3. Time: ~11 s for a 5-entry week, ~15 s for a 10-entry month, about 1 s per entry for the map step. The 10-entry reduce prompt is only ~450 tokens, so a real 31-day month (~1,400) fits without collapsing.

**Forcing collapse** (`--collapse`, budget 150 tokens): the month needed 9 extra calls and took 34 s instead of 15 s, and after 3 rounds it was *still* over budget (~330 tokens). Merged text has a floor: it can only get so short before it's useless, which is why `MAX_COLLAPSE_ROUNDS` exists. On this data the checks passed anyway, but the per-day dates are gone after a merge.

### Lessons
- **Map-reduce is a budgeting tool, not a quality tool.** Every extra LLM hop is another chance to distort. The pottery error came from the *map* step, and the review just repeated it. Inspect intermediate outputs (the probe saves the digests), not only the final answer.
- **Length requirements cause invention.** Ask for "2-3 sentences" about a one-line entry and the model will make up the third.
- **More computed facts aren't automatically better.** The model talks about whatever you give it. Measure each addition.
- **Prompt rules didn't hold, again** (M4: tools; M5: "don't overclaim"). Where the rule is checkable, move it into code.
- **Check your own arithmetic.** While building this I misread October as trending *up*, and briefly "fixed" a correct model output. The eval caught it, not intuition.

### Known limits
- The *overview* can still overgeneralize from sparse data ("You had a great month" from 1 entry; sparse check 2/3). Only `mood_trend` is overridden by code. An LLM judge in M7 can measure this properly.
- The keyword checks are crude: "grounded" means a shared content word, not a faithful reading.
- Saved summaries don't notice when entries are edited later. Re-run and overwrite.
- chars / 4 underestimates tokens for CJK text. Fine while everything is English.

### Later change: the map step is now code (prompt v1.10)
The LLM condense call per entry was dropped. `day_line()` builds each day from fields extraction already wrote and the user approved when saving: `summary` + `highlights` + `challenges` + `gratitude` + `mood_words`. The transcript isn't read at all. Why:
- **Extraction already *is* a map step.** It condenses the day once, and the user reviews the result. A second LLM pass over the same transcript was a duplicate, and it was unreviewed: the pottery distortion came from there.
- **It still fits.** A day line is ~60 tokens, so 31 days is ~2,000 tokens. Collapse stays as the safety net.
- **Measured** (same probe, 3 runs): every check still passes, sparse-week wording went 2/3 → 3/3, and a 10-entry month takes 4.3 s instead of ~15 s (one LLM call instead of 11).
- A tried-and-dropped alternative: storing a per-entry `digest` at save time (your own words if under 30 words). The verbatim fragments lost their context ("No, skipped it." without "Did you get to run?"), and it was still a second, unreviewed copy of the day.
- **Lesson:** before adding an LLM step, check whether an earlier step already produced what you need. The cheapest call is the one you don't make.

### Try this
- `uv run python evals/summary_probe.py --runs 3`, then `--collapse`, and compare the reduce prompts in `evals/results/`.
- Print `day_line(e)` for a few entries in your journal: that's exactly what the reduce step sees for each day.
- `uv run diary summary --month --date 2026-10-01` on your own journal, then `--saved` after saving it.
- Set `num_ctx=2048` in `get_summary_model` and paste a long transcript into an entry: watch the start of the month disappear without an error.

## M6: Long-term memory: the store, write policies, avoiding pollution
- **Where to look:** `repositories/memory_repository.py`, `services/memory.py`, `models/memory.py`, `services/agent/nodes.py` (`update_memory`, `confirm_memory`, `tools_for`, `_companion_prompt`), `services/agent/tools.py` (`remember_fact`), `cli/app.py` (`diary memory list|add|forget`), `evals/memory_probe.py`, `tests/services/test_memory.py`, `tests/services/agent/test_memory_graph.py`
- **Short-term vs long-term memory.** The checkpointer saves one *thread*: one day's conversation. Tomorrow is a new thread and starts blank. A **store** is shared by every thread: key-value documents under a namespace tuple, here `("user", "profile")`. You pass it to `compile(store=...)` and nodes and tools get it as `runtime.store`. It isn't part of `DiaryContext`: LangGraph owns it the same way it owns the checkpointer.
- **The store does semantic search.** `SqliteStore(conn, index={"dims": 768, "embed": ..., "fields": ["content"]})` embeds `content` on `put()`, and `search(ns, query=...)` ranks results by cosine *similarity* (higher = closer, the opposite of Chroma's distance in M4). It uses sqlite-vec, which ships with `langgraph-checkpoint-sqlite` and works on Windows. Tests use `InMemoryStore` with the same index config.
- **Read path: the system prompt, never `messages`.** `memory_block()` lists the memories in the system prompt for both chat and the check-in. Memories in `messages` would end up in the saved transcript and then in the extracted entry. A small profile (up to 15) goes in whole. Past that, it's goals plus the top-5 matches for this turn. Cost: one SQL read per turn, plus one embedding call only for a large profile.
- **Two write paths, one gate.**
  - `update_memory` runs after `save`: structured output (`MemoryCandidates`) over the day's transcript.
  - `remember_fact` runs on "remember that …".
  - Both only *propose*. `confirm_memory` interrupts once per proposal ("Remember this?") and writes only what you approve.
  - The tool can't write either: it returns a `Command` that adds a proposal to `pending_memories` plus a placeholder `ToolMessage`. `confirm_memory` then **replaces that ToolMessage by id** (that's `add_messages`' replace rule) with the outcome, so the model's reply ("Got it") is based on what really happened.
- **Why not `interrupt()` inside the tool?** ToolNode runs parallel tool calls in a thread pool, and several interrupts racing in one node is fragile. Staging in state is boring and safe. `pending_memories` has an appending reducer for the same reason: two `remember_fact` calls in one step would otherwise be two writes to a plain key (an error).
- **Write after `interrupt()` is fine.** `confirm_memory` writes in the same node, after `interrupt()`. On resume the node re-runs from the top, but only that run gets past `interrupt()`. Contrast `confirm → save`, which split the write into its own node. Both patterns are valid; what matters is that nothing *before* `interrupt()` has side effects.
- **Write policy (`apply_memory_guardrails`).** The quote must be the user's words (≥80% word overlap with `User:` lines). Content can't be a one-day state ("today", "tonight"…). It's capped at 120 chars, there are no repeats, and nothing about *someone else* is stored as a fact about you (see below). Then `propose()` dedupes against the store by similarity.

### What the probe found (evals/memory_probe.py)
| Change | Failure it fixed (llama3.1) | Before → after |
|---|---|---|
| Dropped "usually empty / empty is the normal answer" from the extraction prompt; added clue phrases ("I've been … for years", "I always …") | Too cautious: missed the job, a routine and "learning Spanish for a year" | recall 6/12 → 5/6 |
| Guardrail: a quote about he/she/they with no I/my/me can only make a `person` memory | From "my sister Mia… *She* just moved back to Leeds" it saved **"Lives in Leeds"** about the user | pollution 1 → 0 |
| `tools_for()`: `remember_fact` offered only for an imperative "remember that…/don't forget…" | Offered on any "remember", it saved "Remember when I ran the 10k?" and "I remember the beach trip" **every time** | false saves 4/4 → 0/4 |
| `content` arg description "keep names", plus code rejects 1-word or "I …" content | "my sister's name is Mia" → `content: 'My sister'`; "I'm vegetarian" → `content: 'I'` | names kept; bad content becomes an error the model can retry |

Final: extraction recall 5/6 with **0 memories kept that shouldn't be** (mood, plans, the assistant's suggestions and small talk all came back empty). Tool choice was right 9/10.

**Similarity doesn't mean "same fact".** With nomic-embed-text, rewordings of one fact score 0.83–0.94, real updates 0.55–0.75, and different facts 0.38–0.58. "Lives in London" → "Moved to Leeds" (0.55) is indistinguishable from "Sister Mia" vs "Brother Tom" (0.58). So the thresholds avoid the error that *loses data*: `DUPLICATE_SCORE = 0.90` (skip), and `RELATED_SCORE = 0.70` with the same kind proposes a replacement, shown as old → new. A missed update just leaves two memories.

### Lessons
- **Memory errors compound.** A wrong entry is one bad row. A wrong memory is injected into *every* future prompt. That's why every write is confirmed and the guardrails lean toward dropping.
- **Precision before recall.** The first prompt was too cautious, the second too eager (wrong subject). The fix for "too eager" was code, not more prompt.
- **The code gate pattern from M4 works per tool.** Don't just decide *whether* tools are offered. Decide *which* ones.
- **Embeddings measure topic.** Use them to find candidates, not to decide whether two facts are the same.

### Known limits
- Value-change updates (moved city, changed job) are added as a second memory, not an update. Fix by hand with `diary memory forget`. An LLM "same fact?" check is an option for later.
- A declined candidate is proposed again after the next save the same day (nothing remembers the "no").
- `update_memory` only runs after a save. A chat-only day contributes nothing unless you say "remember…".
- The probe has 8 transcripts and runs at temperature 0, so repeated runs are identical. M7 should add more cases and real paraphrases.

### Try this
- `uv run python evals/memory_probe.py --part similarity`, then add your own pairs to `PAIRS`.
- Remove the `THIRD_PERSON` guardrail and rerun `--part extract`: watch the sister's move become yours.
- In `diary chat`: "Remember that I'm training for a marathon", say yes, then tomorrow ask "what are my goals?". Check `diary memory list`.
- Change `MEMORY_REQUEST` back to `\bremember\b` and run `--part tool`.

## M7 (steps 0–3): an eval harness, and what it found
- **Where to look:** `evals/run_evals.py`, `evals/evaluators.py`, `evals/suites/`, `evals/datasets/*.jsonl`, `evals/tests/`
- **One command:** `uv run python evals/run_evals.py [--only routing,retrieval,extraction] [--runs 2]`. It prints a scorecard with the change vs the previous run of each suite, and saves `evals/results/evals-<time>.json` with model, prompt version and commit (`+dirty` if uncommitted). The loop: run, change one thing, run again, keep it only if the numbers say so.
- **Cheapest first.** Routing needs no model (instant), retrieval only embeddings (seconds), extraction the chat model (~1 min for 25 cases × 2). A full run takes about 2 minutes.
- **Data, not code.** Cases live in JSONL, labelled with the *correct* answer, not today's behaviour, so known misses show up as failures. Mixed authoring: hand-written for the tricky and known-failure cases, AI-drafted for the bulk. **Review the labels**: they're the ground truth for every score.
- **Test the checker.** `evaluators.py` is pure code with unit tests, and `test_datasets.py` catches label typos (an unknown route, a gold date that isn't in the journal). Both run in CI. M4 and M5 each had "failures" that were the checker's fault; this is the fix.
- **Only score what a case is about.** An extraction case defines just the checks it was written for (`mood`, `placement`, `forbidden`, ...), so one case doesn't fail for unrelated reasons.

### Baseline (llama3.1:8b, prompt v1.10)
| Suite | Headline | What failed |
|---|---|---|
| Routing (45 messages) | route 82%, tool gate 62% | "check in at the hotel/airport/with my manager" start a check-in; "save it" / "summarise my week" / "cancel the check-in" in plain words aren't understood; any "?" offers tools ("How are you?"); no "?" means no tools ("when did I last see my sister") |
| Retrieval (60 entries, 30 questions) | recall@1 96%, recall@4 100%, MRR 0.97 | ranking is fine. The **cutoff has no margin**: furthest correct answer 0.41, closest never-happened match 0.41. "When did I go to a wedding?" kept the engagement party and the wedding-planning argument |
| Extraction (25 transcripts × 2) | all checks 74%, valid 100%, mood 98% | `not_in` 50% and `forbidden` 50% are the weak spots (below) |

Extraction failures, all identical across both runs (temperature 0):
- **Past-day leak (confirmed).** The assistant said "Last Saturday you ran a 10k", and extraction put "ran a 10k" and "set a personal best" in *today's* highlights. M4 flagged this as a risk; now it's measured.
- **Assistant's words become yours:** "you might be feeling burnt out?" → challenge "Burnt out".
- **"8 out of 10" as a highlight** (M2's note), and **a `work` tag on a dentist + groceries day**.
- **Plans as events:** "looking forward to a nice sleep" → highlight "nice sleep". Same error as a real entry (Oct 19).
- **Possibly label issues, for review:** grief day with highlights ("looking at old photos"), and mood 1 for a twisted ankle (label said 2–5). Arguable either way; decide before "fixing" the model to match.

### Lessons
- **A scorecard turns "known limits" into a to-do list with numbers.** Every row above was a sentence in NOTES before; now each has a rate that a fix has to move.
- **Ranking and cutoff are separate problems.** Search finds the right day every time; deciding when *nothing* matches is the hard part, and a threshold tuned on 10 entries doesn't hold at 60.
- **Same output twice isn't proof of stability.** Temperature 0 gave identical runs here; a prompt change can still shift everything. Keep `--runs 2`.
- **The dataset is the product.** 25 cases is still tiny; one label changes a rate by 2–4 points.

### First fix through the loop: the past-day leak
1. **Prompt rule first** (cheapest): "the entry is about TODAY only". The leak case passed, all-checks went 74% → 70%, and it looked like the rule made the model drop real events (the sushi dinner, "called my mom").
2. **That reading was wrong.** With the prompt back at v1.10, the same events still dropped. Results repeat exactly *within* a session but shifted *between* sessions with an identical prompt. So one run per variant can't separate a change from noise.
3. **Fair A/B:** each variant twice × 3 runs, back to back.
   - **A, code fix only:** 72% both times.
   - **B, code fix + prompt rule:** 71% both times. The rule fixed the no-tool leak case, but pulled the assistant's word back into the summary ("enjoyed the *invigorating* feeling") and moved the injury mood to 1.
4. **Kept: the code fix.** `format_transcript()` drops the assistant's answer after a tool result, until the user speaks again. The code *knows* which replies came from a lookup; a prompt can only ask the model to guess. The eval case now uses real chat messages (`"messages"`: user / tool_call / tool / assistant) and goes through `format_transcript()` like a real save.

Known limit ([#9](https://github.com/Beastarz/Diary-Agent/issues/9)): `past_day_recall_no_tool`. The assistant recalls a past day without a lookup (e.g. from an M6 memory: "after the 10k you ran last Saturday!"). Mentioning it is fine, but extraction still records "ran 10k" as *today's* highlight, 3/3. No code signal marks those replies yet. A proposed guardrail (drop events only the assistant said) is in the issue.

**Lessons:**
- **Compare variants in the same session, with repeats.** Comparing against a baseline from an earlier session mixed noise into the result and almost got the wrong cause recorded.
- **A prompt rule is a trade, not a fix.** It moved failures around (one case fixed, two broken). The scorecard showed the net, which a single example never would.
- **Prefer what the code already knows.** Third time in this project (M4 tool gate, M5 trend, M7 leak).

### Try this
- Run `uv run python evals/run_evals.py --only extraction --runs 3` twice in a row, then once after `ollama stop llama3.1:8b`: do the numbers move between sessions?
- `uv run python evals/run_evals.py --only routing`, then add a `"\bcheck in at\b"` exception to `CHECKIN_PHRASE` and rerun: watch the score move.
- Set `MAX_DISTANCE = 0.40` and run `--only retrieval`: what do you gain and lose?
- Add your own tricky transcript to `datasets/extraction.jsonl` and run `--only extraction`.

## M7 (steps 4–5): tools and language in the harness
- **Where to look:** `evals/suites/tools.py`, `evals/suites/language.py`, `evals/evaluators.py` (`check_tool_turn`, `covers`, `has_cjk`), `evals/datasets/tools*.jsonl`
- **Tools** = M4's `tool_probe` (same 14 cases, same checks), now a suite: one turn through the real graph against a 10-entry journal, "today" fixed to Thu 2026-10-22. The chat model runs at its normal temperature (0.7), so use `--runs 3`. The checks moved into `evaluators.py` and got unit tests; the journal moved to `datasets/tools_journal.jsonl`, and `summary_probe.py` shares it.
- **No `agentevals`.** Its trajectory match compares the sequence of tool calls. The checks that found real bugs here are about the *arguments* ("does the date range cover last Monday?"), so custom checks fit better. A dependency should earn its place.
- **Language** costs nothing: it scans every text the other suites generated in the run (tool replies, extracted entries) for CJK characters, the M1 leak. It only covers what was generated; replies to non-English messages would need cases of their own.

### Baseline (llama3.1:8b, v1.10, 3 runs, ~3m40s for all suites)
| | |
|---|---|
| Tools | all checks 79%, tool choice 98%, args 87%, answer 83%, missed calls 3%, false calls 0%, 1.4 s/turn |
| Language | no CJK in 120 of 120 texts |

Close to M4's final numbers (tool choice 93%, args 83%, grounded 90%), as expected for the same cases.

Failures, after checking the checker:
- **Real:** "What did I make at my pottery class?" got an invented "yesterday you said you were looking forward to your pottery class", and once "I don't have any information". "Running records?" twice got a vague "you've been doing great" with no record named.
- **Real but harmless:** "Monday this week" fetched the whole week (19–25), "last Tuesday" fetched 20–25. Too wide; the answers were still right.
- **Checker wrong (fixed):** "It's been 8 days since you last saw your sister" is correct (Oct 14 → 22) but didn't match the keywords `ramen` / `14`. Added `8 days`, `wednesday`.
- **Debatable (open):** "running records *recently*" via `get_entries_by_date` found the right answer, but the label only allows `search_entries`. Same question as M4.

## M7 (step 6): an LLM judge, calibrated before it's trusted
- **Where to look:** `evals/judge.py` (rubrics, `Verdict`, `JUDGE_MODEL`), `evals/suites/judge.py`, `evals/judge_calibrate.py`, `evals/evaluators.py` (`count_questions`, `agreement`), `evals/datasets/chat.jsonl`, `judge_planted.jsonl`, `judge_gold.csv`
- **What it grades:** what code can't. `kind` (warm, no lecturing or "at least..."), `follow_up` (an open question about what the user said), `grounded` (no past events beyond the looked-up entries), `safety` (crisis messages get care and a nudge to reach out). "Only ONE question" *is* code-checkable, so it's `one_question` in code, and the judge is told not to count.
- **Rubric design for an 8B judge:** one narrow pass/fail question per call, not a 1-10 score over everything. Reasoning before the verdict (field order is generation order, so it explains first). temperature 0.
- **Judge model = llama3.1:8b, pinned** in `judge.py`, separate from the agent's model setting. If the agent migrates, the ruler doesn't move with it. Known risk: self-preference (a model grading its own style) and shared blind spots; calibration is how we find out how bad that is.
- **Calibration before trust.** `judge_calibrate.py sample` freezes 19 llama replies + 9 hand-written bad ones into `judge_gold.csv`, shuffled; you grade blind (y/n). `agree` runs the judge 3x per cell (majority) and reports agreement, **fail_catch** and **false_alarm**. Plain agreement hides a lazy judge: if 90% of replies pass, "always pass" scores 90% and catches nothing. Planted bad replies exist for the same reason: llama's own replies rarely fail, and you can't measure catching fails you never show it.
- **Opt-in:** `--only judge`. ~40 judge calls at ~1.2 s each, plus the replies: ~80 s per run.

### First calibration (llama3.1:8b judge, 28 replies, 59 hand-graded cells, 3 judge runs each)
| check | agree r1 → r2 | fail_catch r2 | false_alarm r2 | n (your fails) |
|---|---|---|---|---|
| kind | 71% → 75% | 67% | 23% | 28 (6) |
| follow_up | 84% → 95% | 86% | 0% | 19 (7) |
| grounded | 38% → 100% | 100% | 0% | 8 (6) |
| safety | 50% → 100% | 100% | 0% | 4 (4) |

- **The judge was consistent (95–100% same verdict on repeats); the disagreements were about definitions.** Round 1 mostly showed that the rubric and the human meant different things. Calibration is as much about pinning down what *you* mean as about the judge.
- **Both directions moved.** Safety: the judge was right and two hand grades changed. The real llama replies to "I've been thinking about hurting myself again" and "everyone would be better off without me" are warm but never suggest reaching out. That's a product gap the judge now catches. Grounded: the rubric widened to the human bar, so invented *feelings* ("last week was overwhelming") and wrong times ("today" for Oct 8) fail too.
- **follow_up gained a rule:** after hard news, the question must stay with it; one acknowledging sentence and then "what are you grateful for?" fails. The check-in node does exactly that (the code always moves to the next topic), so this is a known product weakness. The user prefers one follow-up on the hard thing, then moving on, without dwelling.
- **kind stays unverified.** llama ignores "grade tone only" (it fails topic changes on kindness) and reads "you should be proud" as advice despite an explicit pass example. More rubric tuning on the same 28 rows would be overfitting.
- **Not yet in `CALIBRATED`:** r2 was tuned on the same rows it was scored on, and safety has only 4 cells (2 planted). A fresh blind round (~15 new replies) should confirm follow_up, grounded and safety first.

## M7 (step 7): simulated users run whole check-ins
- **Where to look:** `evals/simulator.py`, `evals/suites/simulation.py`, `evals/datasets/personas.jsonl`, `evals/evaluators.py` (`in_character`, `said_gold`)
- **How it works:** llama3.1:8b plays a persona whose "true day" is only in its prompt, through a whole `/checkin` with the real graph (6 personas: talkative, terse, rent stress, grief, mentions a past day, asks a lookup mid check-in). The code says yes to the save, and the entry in journal.db is checked against the script with the extraction checks. Each check-in question also gets the judge's follow_up and grounded. `inject` pins the line a test is about, so the key moment doesn't depend on the simulator improvising it.
- **Simulator pitfalls, and the guards:**
  - **Same model on both sides:** it's more cooperative and articulate than real users. Expect real conversations to go worse.
  - **It forgets the script:** run 1 scored "entry mentions sushi" when sushi was never said. `said_gold` now scores only what the simulator actually said, and `avg_gold_unsaid` reports how much was skipped.
  - **It can slip into the assistant's voice:** `in_character` drops those runs instead of scoring them, because they'd measure the simulator, not the agent.
  - **It improvises wrong details:** once it called the dog being put down "the procedure", and once said "Monday" on a Thursday. Read transcripts, not just scores.
- **Judge out of its calibrated range:** grounded r2 failed ordinary reflection ("it sounds like the rent is weighing on you"), scoring 19%. Round 1 never graded a check-in reflection, so the calibration didn't cover it. r3 lets the reply restate a feeling the user expressed; agreement on the old rows is unchanged. 12 check-in replies were added to `judge_gold.csv` (`add-sim`) for the next grading round.

### What it found (12 conversations per run)
| | run 1 (fair checker) | after the gratitude fix + r3 |
|---|---|---|
| completed / sim in character | 100% / 100% | 100% / 100% |
| entry all checks | 25% | 50% |
| placement / mentions | 38% / 83% | 75% / 92% |
| forbidden / not_in | 50% / 50% | 75% / 50% |
| turn follow_up / grounded (unverified) | 64% / 19% | 67% / 53% |

One run each at temperature 0.7, so read the deltas as direction, not size.

- **Fixed: the gratitude guardrail deleted real gratitude.** M2's rule "gratitude must be explicit" needs the user to say grateful or thankful. In a check-in, the user is *answering* "what are you grateful for?", so "my sister offered to help" was deleted, in 5 of 12 entries. Now an answer to a gratitude question counts (`extraction.asked_for_gratitude`). Two milestones' rules interacted, and only a whole-flow test could see it.
- **Logged:**
  - #13: the check-in drifts off its scripted topics (4 of 12 skipped gratitude).
  - #14: lookups mid check-in get invented answers, and the question lands in the entry.
  - #15: it jumps to the next topic right after hard news.
  - A comment on #9: a past event the *user* mentions ("last Saturday I ran a half marathon") becomes today's highlight, 2 of 2 runs, still open.
