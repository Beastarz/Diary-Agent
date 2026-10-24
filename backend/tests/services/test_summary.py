"""Map-reduce plumbing with fakes: a RunnableLambda writer and a scripted reducer. No Ollama.
These test the budget logic and what reaches each prompt, not the quality of the writing."""

from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from diary_agent.models.entry import JournalEntry
from diary_agent.models.summary import PeriodSummary
from diary_agent.services import summary as svc
from diary_agent.services.extraction import ExtractionError

REVIEW = PeriodSummary(themes=["work"], wins=["10k"], challenges=[], mood_trend="Up.", overview="You ran.")


def make_entry(day: int, mood: int | None = 6, tags=("work",), transcript="User: fine\nAssistant: ok") -> JournalEntry:
    return JournalEntry(
        entry_date=date(2026, 10, day), mood=mood, mood_words=[], highlights=["h"], challenges=[], gratitude=[],
        tags=list(tags), summary=f"Day {day}.", transcript=transcript, model="fake", prompt_version="test",
    )


def recording_writer(reply=lambda messages: "digest"):
    calls = []

    def fn(messages):
        calls.append(messages)
        return reply(messages)

    return RunnableLambda(fn), calls


def scripted_reducer(*results):
    calls = []

    def fn(messages):
        calls.append(messages)
        return results[len(calls) - 1]

    return RunnableLambda(fn), calls


OK = {"raw": AIMessage(""), "parsed": REVIEW, "parsing_error": None}


@pytest.mark.parametrize(
    "period, day, expected",
    [
        ("week", date(2026, 10, 22), (date(2026, 10, 19), date(2026, 10, 25))),  # Thu -> Mon..Sun
        ("week", date(2026, 10, 19), (date(2026, 10, 19), date(2026, 10, 25))),  # Monday itself
        ("month", date(2026, 10, 22), (date(2026, 10, 1), date(2026, 10, 31))),
        ("month", date(2026, 12, 5), (date(2026, 12, 1), date(2026, 12, 31))),  # year boundary
        ("month", date(2028, 2, 10), (date(2028, 2, 1), date(2028, 2, 29))),  # leap year
    ],
)
def test_period_range(period, day, expected):
    assert svc.period_range(period, day) == expected


def test_stats_are_computed_in_code():
    entries = [make_entry(19, 4, ("work",)), make_entry(20, 8, ("work", "family")), make_entry(21, None, ("family",))]
    text = svc.stats_text(entries, date(2026, 10, 19), date(2026, 10, 25))
    assert text.startswith("Entries: 3 of the 7 days")
    assert "Mood average: 6.0/10, lowest 4 on Monday 19, highest 8 on Tuesday 20" in text
    assert "Tags on 2+ days: work (2 days), family (2 days)" in text
    assert svc.mood_avg([make_entry(1, None)]) is None


def test_entry_text_keeps_user_lines_and_caps_length():
    entry = make_entry(19, transcript="User: " + "a" * 500 + "\nAssistant: how lovely")
    text = svc.entry_text(entry, max_tokens=50)
    assert "Assistant" not in text and "how lovely" not in text
    assert text.endswith(" [...]")
    assert len(text.split("What the user said:\n")[1]) == 50 * svc.CHARS_PER_TOKEN + len(" [...]")


def test_condense_one_call_per_entry_with_code_prefix():
    writer, calls = recording_writer(lambda m: "  You worked.  ")
    digests = svc.condense(writer, [make_entry(19, 4), make_entry(20, None)], "week")
    assert len(calls) == 2
    assert digests == ["Mon 2026-10-19 (mood 4): You worked.", "Tue 2026-10-20 (mood ?): You worked."]


def test_collapse_skipped_when_under_budget():
    writer, calls = recording_writer()
    assert svc.collapse(writer, ["short", "digests"], budget=100) == ["short", "digests"]
    assert calls == []


def test_collapse_merges_groups_in_order_until_it_fits():
    writer, calls = recording_writer(lambda m: "merged")
    digests = [f"day {i} " + "x" * 40 for i in range(6)]  # ~12 tokens each, ~72 total
    out = svc.collapse(writer, digests, budget=30)
    assert svc.estimate_tokens("\n\n".join(out)) <= 30
    first_group = calls[0][1].content
    assert first_group.startswith("day 0") and "day 1" in first_group  # consecutive days stay together


def test_collapse_gives_up_if_model_never_shortens():
    writer, calls = recording_writer(lambda m: "y" * 400)  # always ~100 tokens, over any small budget
    out = svc.collapse(writer, ["a" * 400, "b" * 400], budget=50)
    assert len(calls) <= 2 * svc.MAX_COLLAPSE_ROUNDS
    assert out  # still returns something for the reduce step


def test_summarize_period_end_to_end():
    writer, writer_calls = recording_writer()
    reducer, reducer_calls = scripted_reducer(OK)
    entries = [make_entry(21, 8), make_entry(19, 4)]  # newest first, like the repository
    record = svc.summarize_period(
        writer, reducer, entries, "week", date(2026, 10, 19), date(2026, 10, 25), "fake"
    )
    assert record.themes == ["work"] and record.entry_count == 2 and record.mood_avg == 6.0
    prompt = reducer_calls[0][1].content
    assert prompt.index("Mon 2026-10-19") < prompt.index("Wed 2026-10-21")  # chronological
    assert "Mood average: 6.0/10" in prompt
    assert record.mood_trend == "Too few rated days for a trend (Monday 19 4/10, Wednesday 21 8/10)."


def test_model_writes_trend_with_enough_days():
    reducer, _ = scripted_reducer(OK)
    entries = [make_entry(d) for d in (19, 20, 21)]
    record = svc.summarize_period(
        recording_writer()[0], reducer, entries, "week", date(2026, 10, 19), date(2026, 10, 25), "fake"
    )
    assert record.mood_trend == "Up."


def test_guardrail_drops_mood_as_a_win():
    review = REVIEW.model_copy(update={"wins": ["Good mood", "10k"], "challenges": ["low mood on Tuesday"]})
    out = svc.apply_guardrails(review, [make_entry(d) for d in (19, 20, 21)])
    assert out.wins == ["10k"] and out.challenges == []
    assert out.mood_trend == "Up."


def test_reduce_retries_then_fails():
    bad = {"raw": AIMessage("{}"), "parsed": None, "parsing_error": "themes missing"}
    reducer, calls = scripted_reducer(bad, OK)
    assert svc.reduce(reducer, ["d"], "stats", "week") == REVIEW
    assert "themes missing" in calls[1][-1].content

    reducer, _ = scripted_reducer(bad, bad, bad)
    with pytest.raises(ExtractionError, match="week summary"):
        svc.reduce(reducer, ["d"], "stats", "week")


def test_no_entries_no_llm_call():
    writer, calls = recording_writer()
    assert svc.summarize_period(writer, None, [], "week", date(2026, 10, 19), date(2026, 10, 25), "fake") is None
    assert calls == []
