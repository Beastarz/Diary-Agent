"""The start-of-chat recap: what the last processed day wrote, so unattended writes get seen."""

from datetime import date

import pytest
from langgraph.store.memory import InMemoryStore

from diary_agent.models.entry import ExtractedEntry, JournalEntry
from diary_agent.models.memory import Memory
from diary_agent.models.person import Mention
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.repositories import people_repository as people_repo
from diary_agent.services.recap import recap

D = date(2026, 11, 14)
ENTRY = ExtractedEntry(highlights=[], challenges=[], gratitude=[], mood_words=[], mood=6, tags=[], summary="ok")


@pytest.fixture
def conn(tmp_path):
    return entry_repo.connect(tmp_path / "journal.db")


def save(conn, day):
    entry_repo.save_entry(conn, JournalEntry(**ENTRY.model_dump(), entry_date=day, transcript="", model="f",
                                             prompt_version="t"))


def test_no_entry_means_no_recap(conn):
    assert recap(conn, None) is None


def test_recap_names_the_last_day_and_what_was_written(conn, embeddings):
    store = InMemoryStore(index={"dims": 11, "embed": embeddings, "fields": ["content"]})
    save(conn, date(2026, 11, 10))
    save(conn, D)
    memory_repo.put_memory(store, Memory(id="m1", kind="goal", content="Marathon", evidence="", source_date=D))
    memory_repo.put_memory(store, Memory(id="m2", kind="goal", content="Older", evidence="",
                                         source_date=date(2026, 11, 10)))
    p, _ = people_repo.upsert_person(conn, "Lucas", "friend", [], D)
    people_repo.replace_mentions(conn, D, [Mention(entry_date=D, person_id=p.id, what_happened="met", evidence="")])
    assert recap(conn, store) == (
        "Last entry: Sat 14 Nov. Remembered 1 thing (diary memory list). Noted Lucas (diary people list)."
    )


def test_recap_without_store_or_people(conn):
    save(conn, D)
    assert recap(conn, None) == "Last entry: Sat 14 Nov."
