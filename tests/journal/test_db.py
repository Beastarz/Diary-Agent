from datetime import date

from diary_agent.journal import db
from diary_agent.journal.entry import JournalEntry


def make_entry(day: date, mood: int | None = 7) -> JournalEntry:
    return JournalEntry(
        entry_date=day,
        mood=mood,
        mood_words=["tired", "content"],
        highlights=["dinner with sister"],
        challenges=["long day at work"],
        gratitude=[],
        tags=["family", "work"],
        summary="You had a long day but a nice dinner.",
        transcript="User: long day",
        model="fake",
        prompt_version="test",
    )


def test_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "j.db")
    entry = make_entry(date(2026, 9, 27))
    db.save_entry(conn, entry)

    loaded = db.get_entry(conn, date(2026, 9, 27))
    assert loaded == entry
    assert db.get_entry(conn, date(2026, 9, 26)) is None


def test_null_mood_and_overwrite(tmp_path):
    conn = db.connect(tmp_path / "j.db")
    db.save_entry(conn, make_entry(date(2026, 9, 27), mood=7))
    db.save_entry(conn, make_entry(date(2026, 9, 27), mood=None))

    entries = db.list_entries(conn)
    assert len(entries) == 1
    assert entries[0].mood is None


def test_list_range_newest_first(tmp_path):
    conn = db.connect(tmp_path / "j.db")
    for d in (1, 5, 10, 20):
        db.save_entry(conn, make_entry(date(2026, 9, d)))

    got = db.list_entries(conn, start=date(2026, 9, 5), end=date(2026, 9, 10))
    assert [e.entry_date.day for e in got] == [10, 5]
    assert len(db.list_entries(conn, limit=2)) == 2
