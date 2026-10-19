"""SQLite storage for journal entries. One entry per day (entry_date is the primary key)."""

import json
import sqlite3
from datetime import date
from pathlib import Path

from diary_agent.core.config import settings
from diary_agent.models.entry import JournalEntry

LIST_FIELDS = ("mood_words", "highlights", "challenges", "gratitude", "tags")

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    entry_date     TEXT PRIMARY KEY,
    mood           INTEGER,
    mood_words     TEXT NOT NULL,
    highlights     TEXT NOT NULL,
    challenges     TEXT NOT NULL,
    gratitude      TEXT NOT NULL,
    tags           TEXT NOT NULL,
    summary        TEXT NOT NULL,
    transcript     TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TEXT NOT NULL
)
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or settings.data_dir / "journal.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph runs nodes in worker threads. Safe here because
    # graph steps on one thread run one node at a time, so the connection is never used concurrently.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def save_entry(conn: sqlite3.Connection, entry: JournalEntry) -> None:
    """Insert or replace the entry for entry.entry_date."""
    row = entry.model_dump(mode="json")
    for field in LIST_FIELDS:
        row[field] = json.dumps(row[field])
    cols = ", ".join(row)
    placeholders = ", ".join(f":{c}" for c in row)
    with conn:
        conn.execute(f"INSERT OR REPLACE INTO entries ({cols}) VALUES ({placeholders})", row)


def _to_entry(row: sqlite3.Row) -> JournalEntry:
    data = dict(row)
    for field in LIST_FIELDS:
        data[field] = json.loads(data[field])
    return JournalEntry.model_validate(data)


def get_entry(conn: sqlite3.Connection, day: date) -> JournalEntry | None:
    row = conn.execute("SELECT * FROM entries WHERE entry_date = ?", (day.isoformat(),)).fetchone()
    return _to_entry(row) if row else None


def list_entries(
    conn: sqlite3.Connection, start: date | None = None, end: date | None = None, limit: int | None = None
) -> list[JournalEntry]:
    """Entries in [start, end], newest first."""
    rows = conn.execute(
        "SELECT * FROM entries WHERE entry_date >= ? AND entry_date <= ? ORDER BY entry_date DESC LIMIT ?",
        ((start or date.min).isoformat(), (end or date.max).isoformat(), limit or -1),
    ).fetchall()
    return [_to_entry(r) for r in rows]
