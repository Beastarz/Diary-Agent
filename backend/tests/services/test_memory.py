"""Memory write policy and read path, without Ollama."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from diary_agent.models.memory import Memory, MemoryCandidates, MemoryItem
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.services import memory as memory_service

DAY = date(2026, 10, 20)
TRANSCRIPT = (
    "User: Had dinner with my sister Mia, she just moved near the beach.\n"
    "Assistant: That sounds lovely! Do you see your family often?\n"
    "User: Not really. I'm tired today though."
)


def item(content: str, evidence: str, kind: str = "fact") -> MemoryItem:
    return MemoryItem(evidence=evidence, kind=kind, content=content)


@pytest.fixture
def store(embeddings):
    return InMemoryStore(index={"dims": 11, "embed": embeddings, "fields": ["content"]})


def remember(store, content: str, kind: str = "fact") -> Memory:
    m = Memory(id=memory_repo.new_id(), kind=kind, content=content, evidence=content, source_date=DAY)
    memory_repo.put_memory(store, m)
    return m


# --- guardrails -------------------------------------------------------------------------

def test_guardrails_keep_a_grounded_lasting_fact():
    kept = memory_service.apply_memory_guardrails(
        [item("Sister Mia lives near the beach", "my sister Mia, she just moved near the beach", "person")], TRANSCRIPT
    )
    assert [m.content for m in kept] == ["Sister Mia lives near the beach"]


def test_guardrails_drop_evidence_the_user_never_said():
    # "family" was the assistant's word; "loves" was never said at all.
    kept = memory_service.apply_memory_guardrails([item("Close to family", "I love my family so much")], TRANSCRIPT)
    assert kept == []


def test_guardrails_drop_one_day_states():
    kept = memory_service.apply_memory_guardrails([item("Feeling tired today", "I'm tired today")], TRANSCRIPT)
    assert kept == []


def test_guardrails_drop_facts_about_someone_else():
    # Seen in evals/memory_probe.py: the sister's move became the user's.
    wrong = item("Lives near the beach", "she just moved near the beach")
    right = item("Sister Mia lives near the beach", "she just moved near the beach", "person")
    assert memory_service.apply_memory_guardrails([wrong, right], TRANSCRIPT) == [right]


def test_guardrails_drop_overlong_and_repeated():
    long = item("x" * (memory_service.MAX_CONTENT_CHARS + 1), "my sister Mia")
    a = item("Sister Mia", "my sister Mia", "person")
    b = item("sister mia.", "my sister Mia", "person")
    assert memory_service.apply_memory_guardrails([long, a, b], TRANSCRIPT) == [a]


def test_extract_memories_retries_then_applies_guardrails():
    results = iter([
        {"raw": AIMessage("{bad"), "parsed": None, "parsing_error": "invalid json"},
        {"raw": AIMessage(""), "parsed": MemoryCandidates(memories=[
            item("Sister Mia", "my sister Mia", "person"), item("Feeling tired today", "I'm tired today")]),
         "parsing_error": None},
    ])
    found = memory_service.extract_memories(RunnableLambda(lambda _: next(results)), TRANSCRIPT)
    assert [m.content for m in found] == ["Sister Mia"]


# --- proposals --------------------------------------------------------------------------

def test_new_fact_is_proposed_as_add(store):
    [p] = memory_service.propose(store, [item("Likes to run in the rain", "")], DAY)
    assert p["action"] == "add" and p["replaces"] is None
    assert p["memory"]["source_date"] == "2026-10-20"


def test_near_identical_fact_is_skipped(store):
    remember(store, "Sister lives near the beach", "person")
    assert memory_service.propose(store, [item("Sister lives by the beach", "", "person")], DAY) == []


def test_related_fact_of_same_kind_is_an_update(store):
    old = remember(store, "Training to run a 10k", "goal")
    [p] = memory_service.propose(store, [item("Wants to run with family", "", "goal")], DAY)
    assert p["action"] == "update"
    assert p["memory"]["id"] == old.id and p["replaces"] == "Training to run a 10k"


def test_related_fact_of_another_kind_is_an_add(store):
    remember(store, "Training to run a 10k", "goal")
    [p] = memory_service.propose(store, [item("Wants to run with family", "", "preference")], DAY)
    assert p["action"] == "add"


def test_apply_proposals_writes_and_updates_in_place(store):
    old = remember(store, "Training to run a 10k", "goal")
    proposals = memory_service.propose(store, [item("Wants to run with family", "", "goal")], DAY)
    memory_service.apply_proposals(store, proposals)
    assert [(m.id, m.content) for m in memory_repo.list_memories(store)] == [(old.id, "Wants to run with family")]


# --- read path --------------------------------------------------------------------------

def test_memory_block_is_empty_without_store_or_memories(store):
    assert memory_service.memory_block(None, "hi") == ""
    assert memory_service.memory_block(store, "hi") == ""


def test_small_profile_goes_in_whole(store):
    remember(store, "Has a dog")
    remember(store, "Training for a marathon", "goal")
    block = memory_service.memory_block(store, "anything")
    assert "- (goal) Training for a marathon" in block and "- (fact) Has a dog" in block


def test_large_profile_sends_goals_and_relevant_only(store, monkeypatch):
    monkeypatch.setattr(memory_service, "MAX_INJECTED", 2)
    remember(store, "Training for a marathon", "goal")
    remember(store, "Sister lives near the beach", "person")
    remember(store, "Hates exam season", "preference")
    picked = [m.content for m in memory_service.select_for_prompt(store, "going to the beach with my sister")]
    assert picked == ["Training for a marathon", "Sister lives near the beach"]
