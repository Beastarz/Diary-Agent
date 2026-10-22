"""Tools the chat agent can call. Read-only on purpose: writes go through extract -> confirm -> save.

Tool design for an 8B model:
- Few tools, distinct names, docstrings that say *when* to use each one. The docstring and
  the Annotated arg descriptions are all the model sees.
- Narrow args: plain strings in one fixed date format, validated here, not trusted. search_entries
  had optional date args too; llama3.1 filled them with junk ('0000-00-00'), so they went.
- Bad input comes back as a short error string, not an exception, so the model can retry.
- Short outputs: one compact line per entry, capped. Long tool output drowns a small model.
- Dependencies come from ToolRuntime (the same DiaryContext the nodes get). The model never sees it.
"""

from datetime import date
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from diary_agent.models.entry import JournalEntry
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services.agent.state import DiaryContext

SEARCH_K = 4
# Cosine distance above which a hit counts as unrelated. Calibrated on the 10-entry eval journal
# with nomic-embed-text: relevant top hits <= 0.41, unrelated >= 0.45. Re-check with M7's bigger set.
MAX_DISTANCE = 0.43
MAX_LISTED = 10


def format_entry(e: JournalEntry) -> str:
    line = f"{e.entry_date.isoformat()} ({e.entry_date:%A}), mood {e.mood or '?'}/10: {e.summary}"
    if e.highlights:
        line += f" Highlights: {'; '.join(e.highlights)}."
    if e.challenges:
        line += f" Challenges: {'; '.join(e.challenges)}."
    return line


def _parse(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(f"Error: {name} '{value}' is not a valid date. Use YYYY-MM-DD.") from None


def _parse_range(start_date: str | None, end_date: str | None) -> tuple[date | None, date | None]:
    start, end = _parse(start_date, "start_date"), _parse(end_date, "end_date")
    if start and end and start > end:
        raise ValueError(f"Error: start_date {start} is after end_date {end}.")
    return start, end


@tool
def search_entries(
    query: Annotated[str, "What to look for, in a few words, e.g. 'stress at work' or 'time with my sister'."],
    runtime: ToolRuntime[DiaryContext],
) -> str:
    """Search the user's past journal entries by topic or meaning.
    Use this when the user asks about a person, feeling, activity or event from their past
    and no specific date is given. For questions about a date, use get_entries_by_date instead."""
    ctx = runtime.context
    hits = vector_repo.search(ctx.vectors, query, k=SEARCH_K)
    # Nearest-neighbour search always returns k results, relevant or not. Without a cutoff the
    # model "finds" a ski trip in a pottery entry (seen in evals/tool_probe.py).
    days = [day for day, distance in hits if distance <= MAX_DISTANCE]
    entries = [e for day in days if (e := entry_repo.get_entry(ctx.conn, day))]
    if not entries:
        return "No matching journal entries found. Tell the user you couldn't find anything; do not guess."
    return "\n".join(format_entry(e) for e in entries)


@tool
def get_entries_by_date(
    start_date: Annotated[str, "First day, YYYY-MM-DD."],
    runtime: ToolRuntime[DiaryContext],
    end_date: Annotated[str | None, "Last day, YYYY-MM-DD. Leave out for a single day."] = None,
) -> str:
    """Get the user's journal entries for a specific day or date range.
    Use this when the user asks about a particular day or period, like 'yesterday',
    'last Tuesday' or 'last week'. Take the dates from the date list in your instructions."""
    try:
        start, end = _parse_range(start_date, end_date)
    except ValueError as e:
        return str(e)
    if start is None:
        return "Error: start_date is required. Use YYYY-MM-DD."
    end = end or start
    entries = entry_repo.list_entries(runtime.context.conn, start=start, end=end)
    if not entries:
        span = start.isoformat() if start == end else f"{start} to {end}"
        return f"No journal entries for {span}. Tell the user there's nothing written for that time; do not guess."
    entries = sorted(entries, key=lambda e: e.entry_date)  # oldest first reads like a story
    lines = [format_entry(e) for e in entries[:MAX_LISTED]]
    if len(entries) > MAX_LISTED:
        lines.append(f"({len(entries) - MAX_LISTED} more entries not shown; ask about a shorter range.)")
    return "\n".join(lines)


TOOLS = [search_entries, get_entries_by_date]
