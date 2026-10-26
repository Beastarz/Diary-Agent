# 5. RAG and embeddings

Retrieval-augmented generation: find the relevant past entries, then let the model answer from them. The hard part turned out to be deciding when *nothing* is relevant.

**Where to look:** [`repositories/vector_repository.py`](../../backend/src/diary_agent/repositories/vector_repository.py), [`services/agent/tools.py`](../../backend/src/diary_agent/services/agent/tools.py) (`search_entries`, `MAX_DISTANCE`), [`repositories/memory_repository.py`](../../backend/src/diary_agent/repositories/memory_repository.py), [`evals/suites/retrieval.py`](../../evals/suites/retrieval.py), [`backend/tests/conftest.py`](../../backend/tests/conftest.py)

---

### 1. RAG = retrieve, then generate
- **Concept:** Instead of hoping the model "knows" your data, search for the relevant pieces and put them in front of it.
- **In this repo:** `search_entries(query)` → Chroma returns the closest **dates** → the full entries are read from SQLite → `format_entry` turns them into a compact tool result → the model answers from that.
- **Why it matters:** Without retrieval, the model improvised: after a restart, "what was the best part of my day?" got "the laughter … with your friend". The user never mentioned laughter.

### 2. One source of truth, and an index you can rebuild
- **Concept:** Keep the real data in a normal database and treat the vector store as a derived index you can rebuild.
- **In this repo:** SQLite (`journal.db`) is the truth; Chroma (`data/chroma/`) only maps text → date. `diary reindex` rebuilds Chroma from SQLite. In `nodes.save`, an indexing failure doesn't fail the save; you get "Saved, but not indexed for search. Run 'diary reindex' later."
- **Why it matters:** Vector stores can get out of sync, change format, or need re-embedding when you switch models. If they're only an index, none of that loses data.

### 3. What you embed matters more than how
- **Concept:** Embed clean, dense text that represents what the record is about, not the raw input.
- **In this repo:** `vector_repository.entry_text()` embeds the *extracted* fields (summary, highlights, challenges, tags), not the transcript. One vector per day, `id = date`, so re-saving a day replaces its vector (`upsert_entry`).
- **Why it matters:** Transcripts are full of chit-chat ("thanks!", "how are you?") that drags every entry toward the same generic point in vector space.

### 4. Read your embedding model's instructions
- **Concept:** Some embedding models expect task prefixes on the text.
- **In this repo:** `DOC_PREFIX = "search_document: "` when indexing, `QUERY_PREFIX = "search_query: "` when searching, as nomic-embed-text requires.
- **Why it matters:** It's a one-line detail that quietly changes retrieval quality. It's easy to miss and hard to debug afterwards.

### 5. Nearest-neighbour search always returns *something*, so add a cutoff
- **Concept:** kNN returns the top k results even when none of them are relevant. You need a relevance threshold.
- **In this repo:** `tools.py → MAX_DISTANCE = 0.43` (cosine distance). It was calibrated on 10 entries: relevant matches ≤ 0.41, unrelated ≥ 0.45.
- **Why it matters (bug):** "When did I go skiing?" returned a pottery entry, and the model happily described an invented ski trip to Niseko. With the cutoff, it says "nothing found" (3/3).

### 6. Ranking and cutoff are separate problems
- **Concept:** "Is the right answer at the top?" and "is anything here relevant at all?" need different metrics.
- **In this repo:** The M7 retrieval suite (60 synthetic entries, 30 questions) scores ranking with recall@1 / recall@k / MRR, and scores the cutoff separately with "never happened" questions.
- **Why it matters (eval):** Ranking was excellent (recall@1 96%, MRR 0.97), but the cutoff has **no margin**: the furthest correct answer and the closest wrong match were both at 0.41. A threshold tuned on 10 entries didn't hold at 60.

### 7. Distance vs similarity: check which direction is "closer"
- **Concept:** Some libraries return a *distance* (lower = closer), others a *similarity* (higher = closer).
- **In this repo:** Chroma's `search` returns cosine **distance**. LangGraph's `SqliteStore` (memory, `memory_repository.search`) returns cosine **similarity**. Both docstrings say so.
- **Why it matters:** Mixing the two up flips every threshold. Note the convention right where you use it.

### 8. Similarity measures topic, not "same fact"
- **Concept:** Two texts on the same subject score as similar even if they say different things.
- **In this repo:** Memory dedupe (`services/memory.py → propose`). Measured with nomic-embed-text: rewordings of one fact 0.83–0.94, real updates 0.55–0.75, different facts 0.38–0.58.
- **Why it matters:** "Lives in London" → "Moved to Leeds" (0.55) looked the same as "Sister Mia" vs "Brother Tom" (0.58). No threshold can separate these, so the thresholds are set to avoid the error that *loses data* ([note 7](07-memory.md)). Use embeddings to find *candidates*, not to decide whether two things are the same.

### 9. Test ranking without a model, and evaluate retrieval separately from reasoning
- **Concept:** A fake embedding lets you unit-test search logic. In evals, measure retrieval on its own so you know whether a bad answer came from search or from the model.
- **In this repo:** `backend/tests/conftest.py → KeywordEmbeddings` (bag-of-words vectors), so the ranking and cutoff code is tested without Ollama. `tool_probe.py` includes a model-free retrieval check.
- **Why it matters:** In M4 that check had recall 4/4, which proved the remaining failures were the model's reasoning, not Chroma.

---

### Try it
- Set `MAX_DISTANCE = 1.0` and ask `diary chat` "when did I go skiing?".
- Set `MAX_DISTANCE = 0.40` and run `uv run python evals/run_evals.py --only retrieval`. What do you gain and what do you lose?
- `uv run python evals/memory_probe.py --part similarity`, then add your own pairs to `PAIRS`.

**Full story:** [NOTES.md](../NOTES.md) → M4 (RAG, cutoff), M6 ("Similarity doesn't mean 'same fact'"), M7 (retrieval baseline).
