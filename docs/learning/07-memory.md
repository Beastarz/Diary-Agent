# 7. Long-term memory

How the agent remembers lasting facts about you across days, and why writing memories is much riskier than reading them.

**Where to look:** [`repositories/memory_repository.py`](../../backend/src/diary_agent/repositories/memory_repository.py), [`services/memory.py`](../../backend/src/diary_agent/services/memory.py), [`models/memory.py`](../../backend/src/diary_agent/models/memory.py), [`services/agent/nodes.py`](../../backend/src/diary_agent/services/agent/nodes.py) (`update_memory`, `confirm_memory`, `_companion_prompt`), [`evals/memory_probe.py`](../../evals/memory_probe.py)

---

### 1. Three kinds of "memory", three lifetimes
| Kind | Lives in | Lasts for | Example |
|---|---|---|---|
| Conversation history | `DiaryState.messages` | resent every turn | what you said 5 minutes ago |
| Checkpointer (short-term) | `SqliteSaver` → `checkpoints.db` | one thread = one day | resume today's chat after a restart |
| Store (long-term) | `SqliteStore` → `store.db` | across all threads | "Training for a marathon in March" |

- **Why it matters:** Tomorrow is a new thread and starts blank. Anything that should outlive the day needs the store.

### 2. The LangGraph store: namespaced documents with semantic search
- **Concept:** A store holds key-value documents under a namespace tuple and is shared by every thread. Give it an embedding index and `search(query=...)` ranks results by meaning.
- **In this repo:** `memory_repository.open_store()` → `SqliteStore(conn, index={"dims": 768, "embed": ..., "fields": ["content"]})`, namespace `("user", "profile")`. It's passed to `build_graph(store=...)` → `compile(store=...)` and read as `runtime.store`. Tests use `InMemoryStore` with the same interface.
- **Why it matters:** Like the checkpointer, it's owned by LangGraph, not by your context object. And like every SQLite connection here, it needs `check_same_thread=False` (see [note 3](03-langgraph.md), point 9).

### 3. Read path: memories go into the system prompt, never into `messages`
- **Concept:** Inject memories as background instructions, not as conversation.
- **In this repo:** `services/memory.py → memory_block()` is added to the system prompt in `_companion_prompt()`. With 15 memories or fewer (`MAX_INJECTED`), all of them go in. Past that, it's goals plus the top-5 matches for this turn with similarity ≥ 0.5.
- **Why it matters:** Anything in `messages` ends up in the saved transcript, and then in the extracted entry. A small profile also costs nothing extra: one SQL read, and an embedding call only when the profile is large.

### 4. Two write paths, one human gate
- **Concept:** Memories are written either implicitly (mined from the conversation) or explicitly (the user asks). Both only *propose*; a human confirms.
- **In this repo:** `update_memory` runs after `save` and extracts `MemoryCandidates` from the day's transcript. `remember_fact` runs when the user says "remember that…" and stages a proposal ([note 4](04-tools-and-agents.md), point 8). `confirm_memory` interrupts once per proposal ("Remember this?") and writes only what you approve. You can also manage memories by hand with `diary memory list|add|forget`.
- **Why it matters:** One pipeline, one approval step, no matter where the memory came from.

### 5. Memory pollution is worse than a bad entry
- **Concept:** A wrong entry is one bad row. A wrong memory is injected into **every future prompt** and shapes every future reply.
- **In this repo:** This is why every write is confirmed and the guardrails lean toward dropping candidates. Aim for precision before recall.
- **Why it matters:** The assistant will confidently "know" something false about you indefinitely, and you may not notice.

### 6. A write policy in code
- **Concept:** Encode "what counts as a memory" as deterministic checks that run after the LLM.
- **In this repo:** `services/memory.py → apply_memory_guardrails()`:
  - the `evidence` quote must be ≥ 80% the user's own words (not the assistant's, not invented)
  - no one-day states ("today", "tonight", …: that's an entry, not a memory)
  - content of at most 120 characters, and no duplicates
  - a quote about he/she/they with no I/my/me can only become a `person` memory
- **Why it matters (bug):** From "my sister Mia… *She* just moved back to Leeds", the model saved **"Lives in Leeds"** as a fact about the *user*. The wrong-subject rule took that from 1 to 0.

### 7. Tune the prompt toward recall, then fix over-eagerness in code
- **Concept:** Extraction prompts swing between too cautious and too eager. Loosen the prompt to find the facts, and catch the bad ones with code.
- **In this repo:** `EXTRACT_MEMORY_SYSTEM`. The first version ("usually empty / empty is the normal answer") missed the user's job, a routine, and "learning Spanish for a year": recall 6/12. Removing that wording and adding clue phrases ("I've been … for years", "I always …") got recall to 5/6, and the guardrails kept false memories at 0.
- **Why it matters:** The fix for "too eager" was code, not more prompt.

### 8. Dedupe and update with thresholds chosen by error cost
- **Concept:** When a new fact resembles an old one, decide: skip it, replace the old one, or add a second memory. Since similarity can't reliably tell an update from a different fact ([note 5](05-rag-and-embeddings.md), point 8), choose thresholds that avoid the *expensive* mistake.
- **In this repo:** `services/memory.py → propose()`: similarity ≥ `DUPLICATE_SCORE` (0.90) → skip. ≥ `RELATED_SCORE` (0.70) *and* the same kind → propose an **update**, shown to the user as old → new. Otherwise → **add**.
- **Why it matters:** Overwriting a different fact loses data. A missed update just leaves two memories, which you can fix with `diary memory forget <id>`.

### 9. Memory must never break the main job
- **Concept:** Memory is a secondary feature. If it fails, the core action (saving the entry) must still succeed.
- **In this repo:** `update_memory` catches failures and adds "(Couldn't check for things to remember.)" to the notice. `put_memory` stores the memory without a vector if embedding fails. `propose` treats an embedding failure as "no match".
- **Why it matters:** The same "fail soft" rule as the search index in [note 5](05-rag-and-embeddings.md), point 2.

### Known limits (good exercises)
- A changed value (moved city, new job) becomes a second memory rather than an update.
- A declined candidate is proposed again after the next save on the same day, because nothing remembers the "no".
- `update_memory` only runs after a save, so a chat-only day contributes nothing unless you say "remember…".

---

### Try it
- In `diary chat`: "Remember that I'm training for a marathon". Say yes, then tomorrow ask "what are my goals?". Check `diary memory list`.
- Remove the `THIRD_PERSON` check in `apply_memory_guardrails` and run `uv run python evals/memory_probe.py --part extract`: watch the sister's move become yours.

**Full story:** [NOTES.md](../NOTES.md) → M6.
