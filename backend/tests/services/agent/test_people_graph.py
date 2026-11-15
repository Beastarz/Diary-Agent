"""Relationships through the graph: a save notes people, and a known person reaches the prompt."""

from dataclasses import replace
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from diary_agent.models.entry import ExtractedEntry
from diary_agent.models.person import PersonMention, PersonMentions
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import people_repository as people_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services.agent.graph import build_graph, open_checkpointer, pending_interrupt, thread_config
from diary_agent.services.agent.state import DiaryContext

DAY = date(2026, 11, 15)
CONFIG = thread_config(DAY)
ENTRY = ExtractedEntry(highlights=[], challenges=[], gratitude=[], mood_words=[], mood=7, tags=[],
                       summary="You met Lucas.")
LUCAS = PersonMention(evidence="met Lucas", name="Lucas", relation="friend", facts=["good at math"],
                      what_happened="met")


def structured(parsed):
    return RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": parsed, "parsing_error": None})


@pytest.fixture
def ctx(tmp_path, embeddings, chat_model_cls):
    return DiaryContext(
        chat_model=chat_model_cls(responses=[AIMessage("reply")]),
        extractor=structured(ENTRY),
        conn=entry_repo.connect(tmp_path / "journal.db"),
        vectors=vector_repo.open_store(tmp_path / "chroma", embeddings),
        day=DAY,
        model_name="fake",
        extractors={"relationships": structured(PersonMentions(mentions=[LUCAS]))},
    )


@pytest.fixture
def graph(tmp_path):
    return build_graph(open_checkpointer(tmp_path / "checkpoints.db"))


def test_save_notes_people(graph, ctx):
    graph.invoke({"user_input": "I met Lucas, who is good at math."}, CONFIG, context=ctx)
    state = graph.invoke({"user_input": "/save"}, CONFIG, context=ctx)
    assert pending_interrupt(graph, CONFIG) is None
    assert state["notice"] == "Saved. Noted 1 person: Lucas (new)."
    assert people_repo.get_person(ctx.conn, "Lucas").facts == ["good at math"]


def test_known_person_reaches_the_prompt(graph, ctx, chat_model_cls):
    class Recording(chat_model_cls):
        seen: list = []

        def _generate(self, messages, *args, **kwargs):
            self.seen.append(messages)
            return super()._generate(messages, *args, **kwargs)

    people_repo.upsert_person(ctx.conn, "Lucas", "friend", ["good at math"], date(2026, 11, 10))
    model = Recording(responses=[AIMessage("How is Lucas?")])
    graph.invoke({"user_input": "Lucas called today."}, CONFIG, context=replace(ctx, chat_model=model))
    assert "- Lucas (friend): good at math. Last seen 2026-11-10." in model.seen[-1][0].content
