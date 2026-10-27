"""Eval suites. Each module has run(runs: int) -> dict with:
- "metrics": {name: float}  the scorecard numbers (rates are 0-1)
- "cases":   [dict]         one record per case (and run), saved in the results JSON
- "failures": [str]         short human-readable lines for the failed cases
"""

import json
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def load_jsonl(name: str) -> list[dict]:
    lines = (DATASETS / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.startswith("//")]


def seed_journal(name: str, folder: Path, embeddings=None):
    """A fresh journal.db + Chroma index in `folder` from a dataset of entries, saved and indexed
    with the app's own repository code. Returns (conn, vector store)."""
    from datetime import date

    from diary_agent.models.entry import JournalEntry
    from diary_agent.repositories import entry_repository as entry_repo
    from diary_agent.repositories import vector_repository as vector_repo

    conn = entry_repo.connect(folder / "journal.db")
    store = vector_repo.open_store(folder / "chroma", embeddings)
    for row in load_jsonl(name):
        entry = JournalEntry(
            entry_date=date.fromisoformat(row["date"]), mood=row["mood"], mood_words=row.get("mood_words", []),
            highlights=row["highlights"], challenges=row["challenges"], gratitude=row.get("gratitude", []),
            tags=row["tags"], summary=row["summary"], transcript="", model="eval-seed", prompt_version="eval",
        )
        entry_repo.save_entry(conn, entry)
        vector_repo.upsert_entry(store, entry)
    return conn, store


def real_graph(folder: Path, today):
    """The app's own graph and chat model over tools_journal.jsonl, with "today" fixed so date
    answers are known. No extractor or memory: a chat turn never reaches extraction, and no store =
    no memories. Returns (graph, context, conn); close conn when done."""
    from langgraph.checkpoint.memory import InMemorySaver

    from diary_agent.core.config import settings
    from diary_agent.core.llm import get_chat_model
    from diary_agent.services.agent.graph import build_graph
    from diary_agent.services.agent.state import DiaryContext

    conn, store = seed_journal("tools_journal.jsonl", folder)
    ctx = DiaryContext(chat_model=get_chat_model(), extractor=None, conn=conn, vectors=store,
                       day=today, model_name=settings.chat_model)
    return build_graph(InMemorySaver()), ctx, conn
