# Learning notes

## M0: Chat models, messages, streaming
- **Where to look:** `src/diary_agent/llm.py`, `scripts/smoke.py`
- A LangChain **chat model** (`ChatOllama`) takes a list of **messages** (`SystemMessage`, `HumanMessage`, `AIMessage`) and returns an `AIMessage`.
- `.invoke()` returns the whole reply. `.stream()` yields chunks as they are generated.
- Keep model creation in one factory (`get_chat_model`). The rest of the app depends only on the `BaseChatModel` interface, so switching models or providers is a config change.
- qwen3 is a *reasoning* model. `reasoning=False` turns off its hidden "thinking", which is faster and fine for small talk. Try `DIARY_REASONING=true` later for summaries.
- **Embeddings** turn text into vectors (768 dims for nomic-embed-text). M4 uses them for semantic search.

## M1: LCEL and chat history
- **Where to look:** `src/diary_agent/chat.py`, `cli.py`, `tests/test_chat.py`
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
- Set `DIARY_CHAT_MODEL=llama3.1:8b` (after `ollama pull llama3.1:8b`) and compare.
- Chat for 30+ turns. The history grows without limit. What happens to speed? (Hint: context window. M3/M6 handle trimming.)
