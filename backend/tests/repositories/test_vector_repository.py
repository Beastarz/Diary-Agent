from datetime import date

from diary_agent.models.entry import JournalEntry
from diary_agent.repositories import vector_repository as vector_repo


def make_entry(day: date, summary: str, tags: list[str] | None = None) -> JournalEntry:
    return JournalEntry(
        entry_date=day,
        mood=6,
        mood_words=[],
        highlights=[],
        challenges=[],
        gratitude=[],
        tags=tags or [],
        summary=summary,
        transcript="",
        model="fake",
        prompt_version="test",
    )


def seeded(tmp_path, embeddings):
    store = vector_repo.open_store(tmp_path / "chroma", embeddings)
    vector_repo.upsert_entry(store, make_entry(date(2026, 10, 1), "Deadline stress at work.", ["work"]))
    vector_repo.upsert_entry(store, make_entry(date(2026, 10, 5), "Dinner with my sister.", ["family"]))
    vector_repo.upsert_entry(store, make_entry(date(2026, 10, 9), "Another work deadline.", ["work"]))
    return store


def test_search_ranks_similar_entries_first(tmp_path, embeddings):
    store = seeded(tmp_path, embeddings)
    hits = vector_repo.search(store, "dinner with family", k=3)
    assert hits[0][0] == date(2026, 10, 5)
    assert hits[0][1] < hits[1][1]  # distances ascend


def test_search_date_range_is_inclusive(tmp_path, embeddings):
    store = seeded(tmp_path, embeddings)
    got = vector_repo.search(store, "work deadline", start=date(2026, 10, 5), end=date(2026, 10, 9))
    assert [d for d, _ in got] == [date(2026, 10, 9), date(2026, 10, 5)]
    assert [d for d, _ in vector_repo.search(store, "work", end=date(2026, 10, 1))] == [date(2026, 10, 1)]
    assert [d for d, _ in vector_repo.search(store, "work", start=date(2026, 10, 9))] == [date(2026, 10, 9)]


def test_resave_replaces_the_days_vector(tmp_path, embeddings):
    store = seeded(tmp_path, embeddings)
    vector_repo.upsert_entry(store, make_entry(date(2026, 10, 5), "Rain at the beach."))
    assert vector_repo.count(store) == 3
    assert vector_repo.search(store, "beach rain", k=1)[0][0] == date(2026, 10, 5)


def test_index_persists_on_disk(tmp_path, embeddings):
    seeded(tmp_path, embeddings)
    reopened = vector_repo.open_store(tmp_path / "chroma", embeddings)
    assert vector_repo.count(reopened) == 3


def test_entry_text_skips_empty_fields():
    text = vector_repo.entry_text(make_entry(date(2026, 10, 1), "A quiet day.", ["rest"]))
    assert text == "A quiet day.\nTags: rest"
