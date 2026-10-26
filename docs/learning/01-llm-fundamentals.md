# 1. LLM fundamentals

The building blocks under everything else: how you talk to a chat model, how LangChain composes calls, and the properties of models that surprise people first.

**Where to look:** [`core/llm.py`](../../backend/src/diary_agent/core/llm.py), [`core/config.py`](../../backend/src/diary_agent/core/config.py), [`core/prompts.py`](../../backend/src/diary_agent/core/prompts.py), [`scripts/smoke.py`](../../scripts/smoke.py)

---

### 1. A chat model takes messages and returns a message
- **Concept:** You send a list of messages (`SystemMessage` for instructions, `HumanMessage` for the user, `AIMessage` for earlier replies) and get one `AIMessage` back. `.invoke()` returns the whole reply; `.stream()` yields chunks as they're generated.
- **In this repo:** `core/llm.py → get_chat_model()` builds a `ChatOllama`; `scripts/smoke.py` checks that both `invoke` and `stream` work.
- **Why it matters:** The system message is where all "behaviour" lives: the companion persona, today's date, tool instructions, long-term memories. Later notes show that *what you put into the message list* is the main lever you have, and also the main source of bugs.

### 2. Depend on the interface, and treat the model as config
- **Concept:** Create models in one factory and have the rest of the app depend only on LangChain's `BaseChatModel`. Then switching model or provider is a config change.
- **In this repo:** `get_chat_model(temperature, max_tokens, num_ctx)` and `get_embeddings()`; the model name comes from `DIARY_CHAT_MODEL` in `core/config.py → Settings`.
- **Why it matters:** The project switched from `qwen3:8b` to `llama3.1:8b` after measuring (point 7), and that change touched one setting. It also makes testing easy: tests pass a fake model wherever a real one would go.

### 3. LCEL: `prompt | model | parser`
- **Concept:** LangChain Expression Language chains `Runnable`s with `|`. Each piece (prompt template, model, parser) is a Runnable, and every Runnable gets `invoke`, `stream` and `batch` for free.
- **In this repo:** M1 used `ChatPromptTemplate` + `MessagesPlaceholder("history")` piped into the model. M3 replaced it with a graph, but Runnables are still everywhere: `model.with_structured_output(...)` returns one, and M5 once used `writer.batch()` to run one call per entry in parallel.
- **Why it matters:** `batch()` is LCEL's built-in "map": a free way to parallelise independent calls. (M5 later removed that map step entirely; see [note 6](06-context-and-summaries.md).)

### 4. The model is stateless: "memory" means resending history
- **Concept:** The model remembers nothing between calls. A conversation only "continues" because you send the whole history every time.
- **In this repo:** M1 kept history in a Python list (lost on exit). M3 moved it into `DiaryState.messages`, saved by a checkpointer. M6 added a store for facts that outlive a day. [Note 7](07-memory.md) compares all three.
- **Why it matters:** Every turn resends the whole day, so cost and latency grow with conversation length. Whatever is in the history also gets read by later steps (extraction reads the transcript), so what you put into history has consequences far downstream.

### 5. Embeddings turn text into vectors
- **Concept:** An embedding model maps text to a fixed-length vector. Similar meanings end up close together, which makes semantic search possible.
- **In this repo:** `get_embeddings()` uses `nomic-embed-text` (768 dimensions). It powers entry search (Chroma) and memory search (`SqliteStore`).
- **Why it matters:** Embeddings measure *topic similarity*, not truth or identity. [Note 5](05-rag-and-embeddings.md) shows where that distinction bit us.

### 6. Temperature 0 is not deterministic
- **Concept:** Even at temperature 0, outputs can change between runs because GPU floating-point math isn't bit-exact.
- **In this repo:** The same prompt flipped between "Took partner out for sushi" and "celebrated with partner" (M2). In M7, results were identical *within* a session but shifted *between* sessions with an unchanged prompt.
- **Why it matters:** Never judge a change from one run. Use repeats, and compare variants side by side in the same session ([note 8](08-evals.md)).

### 7. Choose the model with a measurement, not a reputation
- **Concept:** Small models have quirks that only show up on your task. Measure them on your own inputs.
- **In this repo:** qwen3:8b slipped Chinese characters into English replies ("your sister特别 special"). A tiny probe (3 messages × 3 repeats, regex for CJK characters) found qwen3 leaked in 1–3 of 9 replies under every setting, while llama3.1 leaked in 0 of 27. In M4, qwen3 also missed 40% of needed tool calls (llama 7%).
- **Why it matters:** A prompt fix ("reply in the user's language") didn't stop the leak, and neither did changing the sampling settings. The *model* was the problem. Only measuring told us that. This tiny probe was the project's first eval.

### 8. Prompts are code: version them
- **Concept:** Prompts change behaviour as much as code does, so give them a version and record it with every result.
- **In this repo:** `core/prompts.py → PROMPT_VERSION` (now `v1.10`). Every saved entry, summary and eval result records the model name and prompt version.
- **Why it matters:** When a score moves, you need to know which prompt produced it. Without a version you can't compare runs or roll back.

---

### Try it
- Set `DIARY_CHAT_MODEL` to another Ollama model in `backend/.env` and chat. What changes?
- Chat for 30+ turns in one day and watch the reply latency grow. (Why? Point 4.)
- `uv run python scripts/smoke.py` and look at the embedding dimension it prints.

**Full story:** [NOTES.md](../NOTES.md) → M0, M1 ("Observed failure: first eval idea").
