"""Tools called directly, with a stand-in runtime. Covers input validation and output shape."""

from datetime import date
from types import SimpleNamespace

import pytest

from diary_agent.models.entry import JournalEntry
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services.agent import tools


def make_entry(day: date, summary: str, **fields) -> JournalEntry:
    base = dict(mood=6, mood_words=[], highlights=[], challenges=[], gratitude=[], tags=[])
    return JournalEntry(
        entry_date=day, summary=summary, transcript="", model="fake", prompt_version="test", **{**base, **fields}
    )


@pytest.fixture
def runtime(tmp_path, embeddings):
    conn = entry_repo.connect(tmp_path / "journal.db")
    store = vector_repo.open_store(tmp_path / "chroma", embeddings)
    for e in (
        make_entry(date(2026, 10, 13), "Work deadline all day.", challenges=["deadline"]),
        make_entry(date(2026, 10, 14), "Dinner with my sister.", highlights=["dinner with sister"]),
        make_entry(date(2026, 10, 16), "Went for a run in the rain."),
    ):
        entry_repo.save_entry(conn, e)
        vector_repo.upsert_entry(store, e)
    # The tools only touch runtime.context, so a namespace stands in for ToolRuntime.
    return SimpleNamespace(context=SimpleNamespace(conn=conn, vectors=store))


def search(runtime, **args) -> str:
    return tools.search_entries.func(runtime=runtime, **args)


def by_date(runtime, **args) -> str:
    return tools.get_entries_by_date.func(runtime=runtime, **args)


def test_search_returns_best_match_first_with_details(runtime):
    out = search(runtime, query="dinner with sister")
    first = out.splitlines()[0]
    assert first == "2026-10-14 (Wednesday), mood 6/10: Dinner with my sister. Highlights: dinner with sister."


def test_search_drops_unrelated_hits(runtime):
    # Every entry is a nearest neighbour of *something*; the distance cutoff filters them out.
    assert search(runtime, query="beach").startswith("No matching journal entries")
    assert search(runtime, query="rain").splitlines()[0].startswith("2026-10-16")


def test_single_day_lookup(runtime):
    assert by_date(runtime, start_date="2026-10-16").startswith("2026-10-16 (Friday)")


def test_range_lookup_is_oldest_first(runtime):
    out = by_date(runtime, start_date="2026-10-13", end_date="2026-10-16")
    assert [line[:10] for line in out.splitlines()] == ["2026-10-13", "2026-10-14", "2026-10-16"]


def test_empty_day_says_so(runtime):
    assert by_date(runtime, start_date="2026-10-15").startswith("No journal entries for 2026-10-15")


def test_long_range_is_capped(runtime, monkeypatch):
    monkeypatch.setattr(tools, "MAX_LISTED", 2)
    out = by_date(runtime, start_date="2026-10-01", end_date="2026-10-31")
    assert len(out.splitlines()) == 3
    assert "1 more entries not shown" in out


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"start_date": "last tuesday"}, "not a valid date"),
        ({"start_date": "2026-10-16", "end_date": "2026-10-13"}, "is after"),
        ({"start_date": ""}, "start_date is required"),
    ],
)
def test_bad_dates_come_back_as_errors_the_model_can_read(runtime, args, error):
    out = by_date(runtime, **args)
    assert out.startswith("Error:") and error in out


def test_runtime_is_hidden_from_the_model():
    for t in tools.TOOLS:
        assert "runtime" not in t.tool_call_schema.model_json_schema()["properties"]
