"""Pure helpers behind chat_agent: the date cheat-sheet and the tool gate."""

from datetime import date

import pytest

from diary_agent.services.agent.nodes import LOOKUP_REQUEST, calendar_hint


def test_calendar_hint_for_a_thursday():
    lines = calendar_hint(date(2026, 10, 22)).splitlines()
    assert lines[0] == "- today: Thursday 2026-10-22"
    assert lines[1] == "- yesterday: Wednesday 2026-10-21"
    assert lines[2] == "- this week: 2026-10-19 to 2026-10-25"
    assert lines[3] == "- last week: 2026-10-12 to 2026-10-18"
    assert lines[4] == "- this month: 2026-10-01 to 2026-10-22"
    assert "Tuesday 2026-10-20" in lines[5] and "Tuesday 2026-10-13" in lines[5]
    assert "2026-10-09" in lines[5] and "2026-10-08" not in lines[5]  # 13 days back


def test_calendar_hint_on_a_monday_crosses_the_month():
    hint = calendar_hint(date(2026, 11, 2))
    assert "- this week: 2026-11-02 to 2026-11-08" in hint
    assert "- last week: 2026-10-26 to 2026-11-01" in hint


@pytest.mark.parametrize(
    ("text", "offered"),
    [
        ("What did I do yesterday?", True),
        ("when did I last see my sister", False),  # no "?": the cost of a cheap rule, accepted
        ("Tell me about last week", True),
        ("remind me what I wrote on Monday", True),
        ("Today I had a long meeting.", False),
        ("I'm feeling anxious tonight.", False),
        ("thanks, that helps", False),
        ("I want to tell you something", False),  # "tell" only counts at the start
    ],
)
def test_tools_are_only_offered_for_questions_and_requests(text, offered):
    assert bool(LOOKUP_REQUEST.search(text)) is offered
