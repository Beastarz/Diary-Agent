"""The save-time extractor contract.

After an entry is saved, the day's transcript goes through a list of extractors (registry.py).
Each one turns the transcript into records of its own kind: memories, people, later expenses.
This is the app's extension point. Adding a kind of record = one module that builds an
Extractor, plus one line in the registry. Nothing else changes.

An extractor is five pieces:
- name      stable id for reports, notices and tests
- build     model -> the structured-output runnable, built once at startup, not per turn
- extract   (runnable, transcript) -> validated items: LLM call, retries, code guardrails. Never writes.
- apply     (items, deps) -> Report. Writes idempotently: twice for one day = same database.
- describe  Report -> one sentence for the save notice, or "" when nothing changed

There is no human yes/no on this path: it must be able to run with nobody present (a future
scheduled end-of-day job). Code guardrails, evals, and the notice + next-session recap are what
replace the human gate.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.store.base import BaseStore


@dataclass(frozen=True)
class ExtractorDeps:
    """What apply() may write to. Built by the graph node today, by a scheduler later."""

    conn: sqlite3.Connection  # journal.db
    store: BaseStore | None  # long-term memory; None = memory is off
    day: date  # the entry date the records belong to
    model_name: str


@dataclass
class Report:
    name: str
    changed: int = 0
    details: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class Extractor:
    name: str
    build: Callable[[BaseChatModel], Runnable]
    extract: Callable[[Runnable, str], list]
    apply: Callable[[list, ExtractorDeps], Report]
    describe: Callable[[Report], str]
