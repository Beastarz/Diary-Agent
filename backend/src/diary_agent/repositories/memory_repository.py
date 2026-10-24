"""Long-term memory in a LangGraph store (data/store.db).

Short-term vs long-term memory in LangGraph:
- The checkpointer saves one *thread* (one day's conversation). A new day = a new thread = blank.
- A store is shared across threads: key-value documents under a namespace tuple, here
  ("user", "profile"). Compile the graph with store=... and nodes/tools get it as runtime.store.

The store is given an embedding index on `content`, so search(query=...) ranks by meaning
(sqlite-vec under the hood). Every function takes a BaseStore, so tests can pass InMemoryStore.
"""

import sqlite3
import uuid
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from diary_agent.core.config import settings
from diary_agent.core.llm import get_embeddings
from diary_agent.models.memory import Memory

NAMESPACE = ("user", "profile")
LIST_LIMIT = 1000  # store.search() pages; a personal profile never gets near this


def open_store(path: Path | None = None, embeddings: Embeddings | None = None, dims: int = 768) -> SqliteStore:
    """dims must match the embedding model (nomic-embed-text = 768)."""
    path = path or settings.data_dir / "store.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same threading rule as the checkpointer: nodes run in worker threads.
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn, index={"dims": dims, "embed": embeddings or get_embeddings(), "fields": ["content"]})
    store.setup()  # creates tables on first use
    return store


def new_id() -> str:
    return uuid.uuid4().hex[:8]  # short enough to type in 'diary memory forget <id>'


def put_memory(store: BaseStore, memory: Memory) -> None:
    """Insert, or replace the memory with the same id. Embeds `content` (needs Ollama).
    If embedding fails, the memory is still stored, just without a vector (it won't show up
    in semantic search, but list_memories() and the prompt still see it)."""
    value = memory.model_dump(mode="json", exclude={"id"})
    try:
        store.put(NAMESPACE, memory.id, value)
    except Exception:  # Ollama down, etc.
        store.put(NAMESPACE, memory.id, value, index=False)


def get_memory(store: BaseStore, memory_id: str) -> Memory | None:
    item = store.get(NAMESPACE, memory_id)
    return Memory(id=item.key, **item.value) if item else None


def list_memories(store: BaseStore) -> list[Memory]:
    """All memories, newest first."""
    items = store.search(NAMESPACE, limit=LIST_LIMIT)
    return sorted((Memory(id=i.key, **i.value) for i in items), key=lambda m: m.updated_at, reverse=True)


def search(store: BaseStore, query: str, k: int = 5) -> list[tuple[Memory, float]]:
    """(memory, cosine similarity) pairs, most similar first. Higher = closer, unlike Chroma's distance."""
    items = store.search(NAMESPACE, query=query, limit=k)
    return [(Memory(id=i.key, **i.value), i.score) for i in items if i.score is not None]


def delete_memory(store: BaseStore, memory_id: str) -> bool:
    if store.get(NAMESPACE, memory_id) is None:
        return False
    store.delete(NAMESPACE, memory_id)
    return True
