from datetime import date

import pytest

from diary_agent.models.memory import Memory
from diary_agent.repositories import memory_repository as memory_repo

DIMS = 11  # KeywordEmbeddings: 10 vocabulary words + 1


def make(content: str, kind: str = "fact", **fields) -> Memory:
    return Memory(id=memory_repo.new_id(), kind=kind, content=content, evidence=content,
                  source_date=date(2026, 10, 20), **fields)


@pytest.fixture
def store(tmp_path, embeddings):
    return memory_repo.open_store(tmp_path / "store.db", embeddings, dims=DIMS)


def test_put_get_round_trip(store):
    m = make("Sister lives near the beach", kind="person")
    memory_repo.put_memory(store, m)
    assert memory_repo.get_memory(store, m.id) == m


def test_same_id_replaces(store):
    m = make("Training for a 10k run", kind="goal")
    memory_repo.put_memory(store, m)
    memory_repo.put_memory(store, m.model_copy(update={"content": "Training for a half marathon run"}))
    assert [x.content for x in memory_repo.list_memories(store)] == ["Training for a half marathon run"]


def test_search_ranks_by_meaning(store):
    for text in ("Works at a bakery (work)", "Sister lives near the beach", "Likes to run in the rain"):
        memory_repo.put_memory(store, make(text))
    hits = memory_repo.search(store, "rain run", k=2)
    assert hits[0][0].content == "Likes to run in the rain"
    assert hits[0][1] > hits[1][1]  # similarity: higher is closer


def test_survives_reopen(tmp_path, embeddings):
    path = tmp_path / "store.db"
    m = make("Has a dog")
    memory_repo.put_memory(memory_repo.open_store(path, embeddings, dims=DIMS), m)
    assert memory_repo.get_memory(memory_repo.open_store(path, embeddings, dims=DIMS), m.id) == m


def test_put_without_embeddings_still_stores(tmp_path):
    class Down:
        def embed_documents(self, texts):
            raise ConnectionError("ollama is down")

        embed_query = embed_documents

    store = memory_repo.open_store(tmp_path / "store.db", Down(), dims=DIMS)
    memory_repo.put_memory(store, make("Has a dog"))
    assert [m.content for m in memory_repo.list_memories(store)] == ["Has a dog"]


def test_delete(store):
    m = make("Has a dog")
    memory_repo.put_memory(store, m)
    assert memory_repo.delete_memory(store, m.id) is True
    assert memory_repo.delete_memory(store, m.id) is False
    assert memory_repo.list_memories(store) == []
