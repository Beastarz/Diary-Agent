"""The save-time memory extractor: writes without asking, never overwrites, leaves people out."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from diary_agent.models.memory import Memory, MemoryCandidates, MemoryItem
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.services.extractors import memory as mem
from diary_agent.services.extractors.base import ExtractorDeps

DAY = date(2026, 11, 15)
TRANSCRIPT = "User: I'm training for a marathon. Had dinner with my sister Mia.\nAssistant: Nice!"
MARATHON = MemoryItem(evidence="I'm training for a marathon", kind="goal", content="Training for a marathon")
SISTER = MemoryItem(evidence="my sister Mia", kind="person", content="Sister named Mia")


def structured(parsed):
    return RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": parsed, "parsing_error": None})


@pytest.fixture
def store(embeddings):
    return InMemoryStore(index={"dims": 11, "embed": embeddings, "fields": ["content"]})


def deps(store):
    return ExtractorDeps(conn=None, store=store, day=DAY, model_name="fake")


def test_extract_drops_person_items():
    items = mem.extract(structured(MemoryCandidates(memories=[MARATHON, SISTER])), TRANSCRIPT)
    assert [i.content for i in items] == ["Training for a marathon"]


def test_apply_writes_without_asking_and_describes(store):
    report = mem.apply([MARATHON], deps(store))
    assert [m.content for m in memory_repo.list_memories(store)] == ["Training for a marathon"]
    assert report.changed == 1 and mem.describe(report) == "Remembered 1 thing."


def test_apply_adds_instead_of_overwriting_a_related_memory(store):
    memory_repo.put_memory(store, Memory(id="m1", kind="goal", content="Training to run a 10k", evidence="",
                                         source_date=DAY))
    related = MemoryItem(evidence="", kind="goal", content="Wants to run with family")
    mem.apply([related], deps(store))
    contents = sorted(m.content for m in memory_repo.list_memories(store))
    assert contents == ["Training to run a 10k", "Wants to run with family"]


def test_apply_without_a_store_changes_nothing():
    report = mem.apply([MARATHON], deps(None))
    assert report.changed == 0 and report.error is None and mem.describe(report) == ""
