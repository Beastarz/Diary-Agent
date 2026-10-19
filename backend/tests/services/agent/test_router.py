"""The router is a pure function of state, so it's tested directly: no graph, no model."""

import pytest

from diary_agent.services.agent.nodes import CHECKIN_TOPICS, router


@pytest.mark.parametrize(
    ("text", "route"),
    [
        ("/save", "extract"),
        ("/SAVE ", "extract"),
        ("/summary", "summarize"),
        ("/checkin", "checkin"),
        ("Can we do a check-in?", "checkin"),
        ("let's check in", "checkin"),
        ("I went running", "chat"),
        ("checking the news", "chat"),
    ],
)
def test_routes_from_chat(text, route):
    assert router({"user_input": text, "messages": []})["route"] == route


def test_commands_add_no_message():
    assert "messages" not in router({"user_input": "/save", "messages": []})


def test_checkin_answers_stay_in_checkin():
    out = router({"user_input": "pretty good", "messages": [], "mode": "checkin", "checkin_step": 1})
    assert out["route"] == "checkin"
    assert out["messages"][0].content == "pretty good"


def test_final_checkin_answer_goes_to_extract():
    state = {"user_input": "my sister", "messages": [], "mode": "checkin", "checkin_step": len(CHECKIN_TOPICS)}
    out = router(state)
    assert (out["route"], out["mode"]) == ("extract", "chat")
    assert out["messages"][0].content == "my sister"  # the answer is kept before writing up
