"""SQLite storage for the people the user mentions, in journal.db next to the entries.

Two tables:
- people:   one row per person, keyed by a normalised name. Facts accumulate across days.
- mentions: one row per person per day. A re-run of the day's extraction replaces that day's
  rows (replace_mentions), so running it twice never duplicates anything.
"""

import json
import re
import sqlite3
import uuid
from datetime import date, datetime

from diary_agent.models.person import Mention, Person, Relation

MAX_FACTS = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id         TEXT PRIMARY KEY,
    key        TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL,
    relation   TEXT NOT NULL,
    facts      TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen  TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mentions (
    entry_date    TEXT NOT NULL,
    person_id     TEXT NOT NULL,
    what_happened TEXT NOT NULL,
    evidence      TEXT NOT NULL,
    PRIMARY KEY (entry_date, person_id)
);
"""
PERSON_COLS = ("id", "key", "name", "relation", "facts", "first_seen", "last_seen", "updated_at")


# A role in front of a name: "my husband Dev" is Dev. Family titles that are part of how someone is
# named ("Aunt May", "Grandpa Joe") are not here, so they stay in the key.
ROLE_WORDS = {"husband", "wife", "partner", "boyfriend", "girlfriend", "fiance", "fiancee", "brother", "sister",
              "son", "daughter", "cousin", "friend", "best", "colleague", "coworker", "boss", "manager",
              "neighbour", "neighbor", "mum", "mom", "dad", "mother", "father", "flatmate", "roommate"}


def normalise(name: str) -> str:
    """'  My Boss ' -> 'boss', "Lucas's" -> 'lucas', 'my husband Dev' -> 'dev'.
    The key that makes two spellings one person."""
    words = re.sub(r"'s$", "", " ".join(name.lower().split()).removeprefix("my ")).split(" ")
    while len(words) > 1 and words[0] in ROLE_WORDS:
        words = words[1:]
    return " ".join(words)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _to_person(row) -> Person:
    data = {c: row[c] for c in PERSON_COLS}
    data["facts"] = json.loads(data["facts"])
    return Person.model_validate(data)


def _to_mention(row) -> Mention:
    return Mention.model_validate({c: row[c] for c in ("entry_date", "person_id", "what_happened", "evidence")})


def _merge_facts(old: list[str], new: list[str]) -> list[str]:
    merged, seen = [], set()
    for fact in [*old, *new]:
        key = " ".join(fact.lower().split())
        if key and key not in seen:
            seen.add(key)
            merged.append(fact.strip())
    return merged[:MAX_FACTS]


def upsert_person(
    conn: sqlite3.Connection, name: str, relation: Relation, facts: list[str], day: date
) -> tuple[Person, bool]:
    """Create or merge. Returns the stored person and whether it was created.
    Facts are appended without duplicates; "other" never replaces a known relation."""
    ensure_schema(conn)
    key = normalise(name)
    row = conn.execute("SELECT * FROM people WHERE key = ?", (key,)).fetchone()
    if row is None:
        person = Person(id=uuid.uuid4().hex[:8], key=key, name=name.strip(), relation=relation,
                        facts=_merge_facts([], facts), first_seen=day, last_seen=day)
    else:
        old = _to_person(row)
        person = old.model_copy(update={
            "relation": relation if relation != "other" else old.relation,
            "facts": _merge_facts(old.facts, facts),
            "first_seen": min(old.first_seen, day),
            "last_seen": max(old.last_seen, day),
            "updated_at": datetime.now(),
        })
    data = person.model_dump(mode="json")
    data["facts"] = json.dumps(data["facts"])
    with conn:
        conn.execute(
            f"INSERT OR REPLACE INTO people ({', '.join(PERSON_COLS)}) "
            f"VALUES ({', '.join(':' + c for c in PERSON_COLS)})", data)
    return person, row is None


def replace_mentions(conn: sqlite3.Connection, day: date, mentions: list[Mention]) -> None:
    """The day's mentions, replaced as a whole: a re-run leaves exactly what it found."""
    ensure_schema(conn)
    with conn:
        conn.execute("DELETE FROM mentions WHERE entry_date = ?", (day.isoformat(),))
        conn.executemany(
            "INSERT OR REPLACE INTO mentions (entry_date, person_id, what_happened, evidence) "
            "VALUES (:entry_date, :person_id, :what_happened, :evidence)",
            [m.model_dump(mode="json") for m in mentions])


def get_person(conn: sqlite3.Connection, name: str) -> Person | None:
    ensure_schema(conn)
    row = conn.execute("SELECT * FROM people WHERE key = ?", (normalise(name),)).fetchone()
    return _to_person(row) if row else None


def list_people(conn: sqlite3.Connection) -> list[Person]:
    ensure_schema(conn)
    rows = conn.execute("SELECT * FROM people ORDER BY last_seen DESC, name").fetchall()
    return [_to_person(r) for r in rows]


def mentions_for(conn: sqlite3.Connection, person_id: str) -> list[Mention]:
    ensure_schema(conn)
    rows = conn.execute("SELECT * FROM mentions WHERE person_id = ? ORDER BY entry_date DESC", (person_id,))
    return [_to_mention(r) for r in rows.fetchall()]


def mentions_on(conn: sqlite3.Connection, day: date) -> list[tuple[Person, Mention]]:
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT p.*, m.entry_date, m.person_id, m.what_happened, m.evidence FROM mentions m "
        "JOIN people p ON p.id = m.person_id WHERE m.entry_date = ? ORDER BY p.name", (day.isoformat(),))
    return [(_to_person(r), _to_mention(r)) for r in rows.fetchall()]


def delete_person(conn: sqlite3.Connection, name: str) -> bool:
    person = get_person(conn, name)
    if person is None:
        return False
    with conn:
        conn.execute("DELETE FROM mentions WHERE person_id = ?", (person.id,))
        conn.execute("DELETE FROM people WHERE id = ?", (person.id,))
    return True
