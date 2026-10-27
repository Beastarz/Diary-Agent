"""The checkers are code too. If they're wrong, every score is wrong, so they get tests.
No Ollama here: these run in CI with the backend tests."""

from datetime import date

import pytest

from diary_agent.models.entry import ExtractedEntry
from evaluators import check_extraction, check_tool_turn, confusion, covers, has_cjk, matches, mrr, rank_of, recall_at

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


def test_has_cjk():
    assert has_cjk("今天很好") and has_cjk("ありがとう") and has_cjk("감사합니다")
    assert not has_cjk("Thanks, café time! 10k ✓")


DATE_CASE = {"tools": ["get_entries_by_date"], "cover": ["2026-10-19"], "exclude": ["2026-10-12"]}


def call(name, **args):
    return {"name": name, "args": args}


def test_covers_date_ranges():
    assert covers(call("get_entries_by_date", start_date="2026-10-19"), DATE_CASE)  # one day
    assert covers(call("get_entries_by_date", start_date="2026-10-18", end_date="2026-10-20"), DATE_CASE)
    assert not covers(call("get_entries_by_date", start_date="2026-10-12", end_date="2026-10-19"), DATE_CASE)
    assert not covers(call("get_entries_by_date", start_date="0000-00-00"), DATE_CASE)  # M4's junk date
    assert not covers(call("search_entries", query="monday"), DATE_CASE)  # no dates never covers
    any_case = {"cover_any": ["2026-10-13", "2026-10-20"]}
    assert covers(call("get_entries_by_date", start_date="2026-10-13"), any_case)


def test_check_tool_turn_split_by_kind():
    ok = check_tool_turn(DATE_CASE, [call("get_entries_by_date", start_date="2026-10-19")], "You worked late.")
    assert ok == {"tool": True, "args": True, "clean": True}
    wrong_tool = check_tool_turn(DATE_CASE, [call("search_entries", query="monday")], "x")
    assert not wrong_tool["tool"] and not wrong_tool["args"]
    assert not check_tool_turn(DATE_CASE, [], "x")["tool"]  # missed call

    none_case = {"tools": []}
    assert check_tool_turn(none_case, [], "Hi!") == {"tool": True, "clean": True}
    assert not check_tool_turn(none_case, [call("search_entries", query="hi")], "Hi!")["tool"]  # false call


def test_check_tool_turn_answers():
    topic = {"tools": ["search_entries"], "answer": ["ramen"]}
    assert check_tool_turn(topic, [call("search_entries", query="sister")], "Ramen on the 14th!")["answer"]
    assert not check_tool_turn(topic, [call("search_entries", query="")], "Sushi.")["args"]
    missing = {"tools": ["search_entries"], "admits": True}
    assert check_tool_turn(missing, [call("search_entries", query="ski")], "I couldn't find any skiing.")["answer"]
    assert not check_tool_turn(missing, [call("search_entries", query="ski")], "You skied in Niseko!")["answer"]
