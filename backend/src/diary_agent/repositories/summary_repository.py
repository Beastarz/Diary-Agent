"""SQLite storage for summaries, in journal.db next to entries. One summary per (period, start_date)."""

import json
import sqlite3
from datetime import date

from diary_agent.models.summary import Period, SummaryRecord

LIST_FIELDS = ("themes", "wins", "challenges")

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    period         TEXT NOT NULL,
    start_date     TEXT NOT NULL,
    end_date       TEXT NOT NULL,
    entry_count    INTEGER NOT NULL,
    mood_avg       REAL,
    themes         TEXT NOT NULL,
    wins           TEXT NOT NULL,
    challenges     TEXT NOT NULL,
    mood_trend     TEXT NOT NULL,
    overview       TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (period, start_date)
)
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Uses entry_repository.connect()'s connection; the table is created on first use."""
    conn.execute(SCHEMA)


def save_summary(conn: sqlite3.Connection, summary: SummaryRecord) -> None:
    """Insert or replace the summary for (summary.period, summary.start_date)."""
    ensure_schema(conn)
    row = summary.model_dump(mode="json")
    for field in LIST_FIELDS:
        row[field] = json.dumps(row[field])
    cols = ", ".join(row)
    placeholders = ", ".join(f":{c}" for c in row)
    with conn:
        conn.execute(f"INSERT OR REPLACE INTO summaries ({cols}) VALUES ({placeholders})", row)


def get_summary(conn: sqlite3.Connection, period: Period, start: date) -> SummaryRecord | None:
    ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM summaries WHERE period = ? AND start_date = ?", (period, start.isoformat())
    ).fetchone()
    if not row:
        return None
    data = dict(row)
    for field in LIST_FIELDS:
        data[field] = json.loads(data[field])
    return SummaryRecord.model_validate(data)
