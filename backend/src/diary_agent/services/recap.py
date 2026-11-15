"""The start-of-chat recap: what the most recent saved day wrote, with the commands that undo it.

The save path has no yes/no any more, so this is how a wrong memory or person gets noticed
within a day. It is computed from the repositories, not from a log table.
"""

import sqlite3

from langgraph.store.base import BaseStore

from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.repositories import people_repository as people_repo
from diary_agent.services.memory import remembered


def recap(conn: sqlite3.Connection, store: BaseStore | None) -> str | None:
    latest = entry_repo.list_entries(conn, limit=1)
    if not latest:
        return None
    day = latest[0].entry_date
    text = f"Last entry: {day:%a %d %b}."
    n = sum(m.source_date == day for m in memory_repo.list_memories(store)) if store is not None else 0
    if n:
        text += f" {remembered(n)[:-1]} (diary memory list)."
    names = [p.name for p, _ in people_repo.mentions_on(conn, day)]
    if names:
        text += f" Noted {', '.join(names)} (diary people list)."
    return text
