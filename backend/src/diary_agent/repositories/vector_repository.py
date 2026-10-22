"""Chroma index of journal entries, for semantic search. One vector per day.

SQLite (entry_repository) stays the source of truth: this index only maps text -> entry dates,
and can always be rebuilt from SQLite (`diary reindex`). Callers fetch full entries from SQLite.
"""

from datetime import date
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from diary_agent.core.config import settings
from diary_agent.core.llm import get_embeddings
from diary_agent.models.entry import JournalEntry

COLLECTION = "entries"
# nomic-embed-text was trained with task prefixes; documents and queries embed differently.
# Other embedding models ignore them harmlessly.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def open_store(path: Path | None = None, embeddings: Embeddings | None = None) -> Chroma:
    path = path or settings.data_dir / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(path),
        client_settings=chromadb.config.Settings(anonymized_telemetry=False),  # no cloud calls
        collection_metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
    )


def _day_key(day: date) -> int:
    """Chroma's $gte/$lte filters only compare numbers, so dates are stored as 20261022."""
    return day.year * 10000 + day.month * 100 + day.day


def entry_text(entry: JournalEntry) -> str:
    """What gets embedded: the extracted fields, not the raw transcript (chit-chat is noise)."""
    parts = [entry.summary]
    for label, items in (
        ("Highlights", entry.highlights),
        ("Challenges", entry.challenges),
        ("Grateful for", entry.gratitude),
        ("Feelings", entry.mood_words),
        ("Tags", entry.tags),
    ):
        if items:
            parts.append(f"{label}: {', '.join(items)}")
    return "\n".join(parts)


def upsert_entry(store: Chroma, entry: JournalEntry) -> None:
    """id = date, so re-saving a day replaces its vector (Chroma's add_texts upserts)."""
    store.add_texts(
        [DOC_PREFIX + entry_text(entry)],
        metadatas=[{"day": _day_key(entry.entry_date)}],
        ids=[entry.entry_date.isoformat()],
    )


def search(
    store: Chroma, query: str, k: int = 5, start: date | None = None, end: date | None = None
) -> list[tuple[date, float]]:
    """Most similar entries as (entry_date, distance), closest first. Optional date range is inclusive."""
    bounds = []
    if start:
        bounds.append({"day": {"$gte": _day_key(start)}})
    if end:
        bounds.append({"day": {"$lte": _day_key(end)}})
    where = bounds[0] if len(bounds) == 1 else ({"$and": bounds} if bounds else None)
    hits = store.similarity_search_with_score(QUERY_PREFIX + query, k=k, filter=where)
    return [(date.fromisoformat(doc.id), distance) for doc, distance in hits]


def count(store: Chroma) -> int:
    return len(store.get(include=[])["ids"])
