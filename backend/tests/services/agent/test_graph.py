"""The graph end to end with fakes: a canned chat model, a scripted extractor, temp SQLite files.
No Ollama. These test the *plumbing* (routing, checkpoints, interrupts), not reply quality."""

from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from diary_agent.models.entry import ExtractedEntry
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.services.agent.graph import (
    build_graph,
    open_checkpointer,
    pending_interrupt,
    stream_turn,
    thread_config,
)
from diary_agent.services.agent.nodes import CHECKIN_TOPICS
from diary_agent.services.agent.state import DiaryContext

DAY = date(2026, 10, 19)
GOOD = ExtractedEntry(
    highlights=["dinner"], challenges=[], gratitude=[], mood_words=[], mood=7, tags=["food"], summary="You ate well."
)


@pytest.fixture
def ctx(tmp_path):
    extractor = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": GOOD, "parsing_error": None})
    return DiaryContext(
        chat_model=FakeListChatModel(responses=["reply"]),
        extractor=extractor,
        conn=entry_repo.connect(tmp_path / "journal.db"),
        day=DAY,
        model_name="fake",
    )


@pytest.fixture
def graph(tmp_path):
    return build_graph(open_checkpointer(tmp_path / "checkpoints.db"))


CONFIG = thread_config(DAY)


def say(graph, ctx, text):
    return graph.invoke({"user_input": text}, CONFIG, context=ctx)


def test_chat_turn_appends_messages(graph, ctx):
    state = say(graph, ctx, "Long day at work.")
    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert state["messages"][1].content == "reply"


def test_conversation_survives_restart(tmp_path, ctx):
    say(build_graph(open_checkpointer(tmp_path / "cp.db")), ctx, "Hello")
    # A brand-new graph + saver on the same file = restarting the CLI.
    restarted = build_graph(open_checkpointer(tmp_path / "cp.db"))
    state = say(restarted, ctx, "Still here?")
    assert [m.content for m in state["messages"] if m.type == "human"] == ["Hello", "Still here?"]


def test_threads_are_separate_days(graph, ctx):
    say(graph, ctx, "Today")
    other = graph.get_state(thread_config(date(2026, 10, 18))).values
    assert other.get("messages", []) == []


def test_commands_do_not_enter_the_conversation(graph, ctx):
    state = say(graph, ctx, "/save")
    assert state["messages"] == []
    assert "Nothing to save" in state["notice"]
    assert pending_interrupt(graph, CONFIG) is None


def test_save_pauses_for_confirmation_then_writes(graph, ctx):
    say(graph, ctx, "Had a great dinner.")
    say(graph, ctx, "/save")

    payload = pending_interrupt(graph, CONFIG)
    assert payload == {"draft": GOOD.model_dump(mode="json"), "exists": False}
    assert entry_repo.get_entry(ctx.conn, DAY) is None  # nothing written while paused

    state = graph.invoke(Command(resume=True), CONFIG, context=ctx)
    saved = entry_repo.get_entry(ctx.conn, DAY)
    assert saved.summary == "You ate well."
    assert saved.transcript == "User: Had a great dinner.\nAssistant: reply"
    assert state["notice"] == "Saved."
    assert state["saved_count"] == len(state["messages"])


def test_declining_does_not_write(graph, ctx):
    say(graph, ctx, "Had a great dinner.")
    say(graph, ctx, "/save")
    state = graph.invoke(Command(resume=False), CONFIG, context=ctx)
    assert entry_repo.get_entry(ctx.conn, DAY) is None
    assert state["notice"] == "Not saved."
    assert state["draft"] is None


def test_second_save_reports_existing_entry(graph, ctx):
    say(graph, ctx, "Had a great dinner.")
    say(graph, ctx, "/save")
    graph.invoke(Command(resume=True), CONFIG, context=ctx)
    say(graph, ctx, "/save")
    assert pending_interrupt(graph, CONFIG)["exists"] is True


def test_checkin_asks_each_topic_then_asks_to_save(graph, ctx):
    state = say(graph, ctx, "/checkin")
    assert state["mode"] == "checkin"
    assert state["messages"][0].content == "Let's do a check-in."
    for i in range(len(CHECKIN_TOPICS) - 1):
        state = say(graph, ctx, f"answer {i}")
        assert pending_interrupt(graph, CONFIG) is None
    assert state["checkin_step"] == len(CHECKIN_TOPICS)

    state = say(graph, ctx, "last answer")  # answers the final question -> extract -> confirm
    assert state["mode"] == "chat"
    assert pending_interrupt(graph, CONFIG) is not None
    # One opening line + one answer per question, each with one reply, except the last answer.
    assert [m.type for m in state["messages"]].count("ai") == len(CHECKIN_TOPICS)


def test_cancel_leaves_checkin(graph, ctx):
    say(graph, ctx, "let's check in")
    state = say(graph, ctx, "/cancel")
    assert state["mode"] == "chat"
    state = say(graph, ctx, "just chatting")
    assert state["route"] == "chat"


def test_save_through_stream_reaches_confirmation(graph, ctx):
    # Regression: the CLI path (stream) runs nodes in worker threads. A default sqlite3
    # connection raised "SQLite objects created in a thread can only be used in that same thread".
    list(stream_turn(graph, {"user_input": "Had a great dinner."}, CONFIG, ctx))
    events = list(stream_turn(graph, {"user_input": "/save"}, CONFIG, ctx))
    assert ("status", "Writing up your entry...") in events
    assert pending_interrupt(graph, CONFIG) is not None
    list(stream_turn(graph, Command(resume=True), CONFIG, ctx))
    assert entry_repo.get_entry(ctx.conn, DAY) is not None


def test_stream_turn_yields_reply_tokens_once(graph, ctx):
    events = list(stream_turn(graph, {"user_input": "hi"}, CONFIG, ctx))
    assert "".join(text for kind, text in events if kind == "token") == "reply"
