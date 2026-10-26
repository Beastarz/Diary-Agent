"""The checkers are code too. If they're wrong, every score is wrong, so they get tests.
No Ollama here: these run in CI with the backend tests."""

from datetime import date

import pytest

from diary_agent.models.entry import ExtractedEntry
from evaluators import check_extraction, confusion, matches, mrr, rank_of, recall_at

D = date(2026, 10, 1)


def entry(**fields) -> ExtractedEntry:
    base = dict(highlights=[], challenges=[], gratitude=[], mood_words=[], mood=6, tags=[], summary="You had a day.")
    return ExtractedEntry(**{**base, **fields})


def test_matches_is_case_insensitive_with_alternatives():
    assert matches("Got PROMOTED", "promot")
    assert matches("dinner with mom", "mother|mom")
    assert not matches("dinner with dad", "mother|mom")


def test_confusion_counts_expected_vs_predicted():
    assert confusion([("chat", "chat"), ("chat", "checkin"), ("extract", "chat")]) == {
        "chat": {"chat": 1, "checkin": 1}, "extract": {"chat": 1}}


def test_rank_recall_and_mrr():
    ranked = [date(2026, 10, d) for d in (5, 3, 1)]
    assert rank_of({date(2026, 10, 1), date(2026, 10, 3)}, ranked) == 2  # first gold hit wins
    assert rank_of({date(2026, 10, 9)}, ranked) is None
    ranks = [1, 2, None, 4]
    assert recall_at(ranks, 1) == 0.25 and recall_at(ranks, 4) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_mood_range_and_null():
    assert check_extraction({"mood": [5, 7]}, entry(mood=6))["mood"]
    assert not check_extraction({"mood": [5, 7]}, entry(mood=8))["mood"]
    assert not check_extraction({"mood": [5, 7]}, entry(mood=None))["mood"]
    assert check_extraction({"mood": [5, 7], "null_ok": True}, entry(mood=None))["mood"]


def test_only_defined_checks_are_scored():
    assert set(check_extraction({"mood": [1, 10]}, entry())) == {"mood"}


def test_placement_and_not_in_look_at_one_field():
    e = entry(highlights=["8 out of 10", "sushi"], challenges=["missed the bus"])
    case = {"mood": [1, 10], "placement": {"challenges": ["bus"]}, "not_in": {"highlights": ["out of 10"]}}
    res = check_extraction(case, e)
    assert res["placement"] and not res["not_in"]
    assert not check_extraction({"mood": [1, 10], "placement": {"highlights": ["bus"]}}, e)["placement"]


def test_forbidden_searches_every_field_including_tags_and_mood_words():
    assert not check_extraction({"mood": [1, 10], "forbidden": ["burn"]}, entry(mood_words=["burnt"]))["forbidden"]
    assert not check_extraction({"mood": [1, 10], "forbidden": ["run"]}, entry(tags=["running"]))["forbidden"]
    assert check_extraction({"mood": [1, 10], "forbidden": ["run"]}, entry(highlights=["read"]))["forbidden"]


def test_tags_any_and_not():
    case = {"mood": [1, 10], "tags_any": ["health|dental"], "tags_not": ["work"]}
    assert check_extraction(case, entry(tags=["health", "errands"]))["tags"]
    assert not check_extraction(case, entry(tags=["health", "work"]))["tags"]
    assert not check_extraction(case, entry(tags=["errands"]))["tags"]
    assert check_extraction({"mood": [1, 10], "tags_not": ["work"]}, entry(tags=["errands"]))["tags"]


def test_empty_and_mentions():
    e = entry(highlights=["ramen with sister"], summary="Work was draining.")
    res = check_extraction({"mood": [1, 10], "empty": ["gratitude"], "mentions": ["sister", "work"]}, e)
    assert res == {"mood": True, "empty": True, "mentions": True}
    assert not check_extraction({"mood": [1, 10], "empty": ["highlights"]}, e)["empty"]
