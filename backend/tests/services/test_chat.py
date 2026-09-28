"""Unit tests use a fake model so they run without Ollama."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from diary_agent.services.chat import ChatSession


def test_session_records_history():
    session = ChatSession(FakeListChatModel(responses=["That sounds tiring.", "Glad to hear!"]))

    reply1 = "".join(session.stream("Long day at work."))
    reply2 = "".join(session.stream("But dinner was great."))

    assert reply1 == "That sounds tiring."
    assert reply2 == "Glad to hear!"
    assert [m.type for m in session.history] == ["human", "ai", "human", "ai"]
    assert session.history[2].content == "But dinner was great."
