"""Memory through the graph: remember_fact (confirmed), the save-time memory extractor
(unattended), and memory in the prompt.
Same fakes as test_graph.py, plus an in-memory store compiled into the graph."""

from dataclasses import replace
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from diary_agent.models.entry import ExtractedEntry
from diary_agent.models.memory import Memory, MemoryCandidates, MemoryItem
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services.agent.graph import (
    build_graph,
    open_checkpointer,
    pending_interrupt,
    stream_turn,
    thread_config,
)
from diary_agent.services.agent.state import DiaryContext

DAY = date(2026, 10, 19)
CONFIG = thread_config(DAY)
ENTRY = ExtractedEntry(
    highlights=["dinner"], challenges=[], gratitude=[], mood_words=[], mood=7, tags=[], summary="You ate well."
)
MARATHON = MemoryItem(evidence="I'm training for a marathon", kind="goal", content="Training for a marathon")


def tool_call(name: str, call_id: str = "c1", **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def structured(parsed):
    return RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": parsed, "parsing_error": None})


@pytest.fixture
def store(embeddings):
    return InMemoryStore(index={"dims": 11, "embed": embeddings, "fields": ["content"]})


@pytest.fixture
def make_ctx(tmp_path, embeddings, chat_model_cls):
    conn = entry_repo.connect(tmp_path / "journal.db")
    vectors = vector_repo.open_store(tmp_path / "chroma", embeddings)

    def make(script=None, memories=(), day=DAY, chat_model=None):
        return DiaryContext(
            chat_model=chat_model or chat_model_cls(responses=script or [AIMessage("reply")]),
            extractor=structured(ENTRY),
            conn=conn,
            vectors=vectors,
            day=day,
            model_name="fake",
            extractors={"memory": structured(MemoryCandidates(memories=list(memories)))},
        )

    return make


@pytest.fixture
def graph(tmp_path, store):
    return build_graph(open_checkpointer(tmp_path / "checkpoints.db"), store)


def say(graph, ctx, text, config=CONFIG):
    return graph.invoke({"user_input": text}, config, context=ctx)


def resume(graph, ctx, answer, config=CONFIG):
    return graph.invoke(Command(resume=answer), config, context=ctx)


REMEMBER = [tool_call("remember_fact", content="Training to run a marathon", kind="goal"), AIMessage("Got it.")]


def test_remember_fact_asks_before_writing(graph, store, make_ctx):
    ctx = make_ctx(REMEMBER)
    say(graph, ctx, "Remember that I'm training to run a marathon.")
    [proposal] = pending_interrupt(graph, CONFIG)["proposals"]
    assert proposal["memory"]["content"] == "Training to run a marathon"
    assert memory_repo.list_memories(store) == []  # nothing written while paused

    state = resume(graph, ctx, [True])
    assert [m.content for m in memory_repo.list_memories(store)] == ["Training to run a marathon"]
    assert [m.type for m in state["messages"]] == ["human", "ai", "tool", "ai"]
    assert state["messages"][2].content == "Saved to memory: Training to run a marathon"  # placeholder replaced
    assert state["messages"][-1].content == "Got it."
    assert state["notice"] == "Remembered 1 thing."


def test_declined_memory_is_not_written(graph, store, make_ctx):
    ctx = make_ctx(REMEMBER)
    say(graph, ctx, "Remember that I'm training to run a marathon.")
    state = resume(graph, ctx, [False])
    assert memory_repo.list_memories(store) == []
    assert "chose not to save" in state["messages"][2].content
    assert state["notice"] == "Nothing remembered."


def test_known_fact_skips_confirmation(graph, store, make_ctx):
    memory_repo.put_memory(store, Memory(id="m1", kind="goal", content="Training to run a marathon",
                                         evidence="", source_date=DAY))
    state = say(graph, make_ctx(REMEMBER), "Remember that I'm training to run a marathon.")
    assert pending_interrupt(graph, CONFIG) is None
    assert state["messages"][2].content.startswith("Already remembered")


def test_save_writes_lasting_facts_without_asking(graph, store, make_ctx):
    ctx = make_ctx(memories=[MARATHON])
    say(graph, ctx, "I'm training for a marathon.")
    state = say(graph, ctx, "/save")
    assert pending_interrupt(graph, CONFIG) is None
    assert entry_repo.get_entry(ctx.conn, DAY) is not None
    assert [m.content for m in memory_repo.list_memories(store)] == ["Training for a marathon"]
    assert state["notice"] == "Saved. Remembered 1 thing."
    assert state["messages"][-1].type == "ai"  # no extra reply after a save-time memory write


def test_save_drops_person_memories(graph, store, make_ctx):
    sister = MemoryItem(evidence="my sister Mia", kind="person", content="Sister named Mia")
    ctx = make_ctx(memories=[sister])
    say(graph, ctx, "Had dinner with my sister Mia.")
    state = say(graph, ctx, "/save")
    assert memory_repo.list_memories(store) == []
    assert state["notice"] == "Saved."


def test_save_drops_ungrounded_candidates(graph, store, make_ctx):
    invented = MemoryItem(evidence="I adore my cat", kind="fact", content="Has a cat")
    ctx = make_ctx(memories=[invented])
    say(graph, ctx, "Had dinner with my sister Mia.")
    state = say(graph, ctx, "/save")
    assert memory_repo.list_memories(store) == []
    assert state["notice"] == "Saved."


def test_memory_failure_does_not_undo_the_save(graph, make_ctx):
    def broken(_):
        raise ConnectionError("ollama is down")

    ctx = replace(make_ctx(), extractors={"memory": RunnableLambda(broken)})
    say(graph, ctx, "Had dinner with my sister Mia.")
    state = say(graph, ctx, "/save")
    assert entry_repo.get_entry(ctx.conn, DAY) is not None
    assert state["notice"] == "Saved. (Couldn't check for memory.)"


def test_save_status_is_streamed_when_extractors_exist(graph, make_ctx):
    ctx = make_ctx(memories=[])
    list(stream_turn(graph, {"user_input": "Hi"}, CONFIG, ctx))
    events = list(stream_turn(graph, {"user_input": "/save"}, CONFIG, ctx))
    assert ("status", "Noting people and things to remember...") in events


def test_memories_reach_the_prompt_on_another_day(graph, store, make_ctx, chat_model_cls):
    """The point of a store: a fact from one thread shows up in a brand-new thread."""
    class Recording(chat_model_cls):
        seen: list = []

        def _generate(self, messages, *args, **kwargs):
            self.seen.append(messages)
            return super()._generate(messages, *args, **kwargs)

    ctx = make_ctx(REMEMBER)
    say(graph, ctx, "Remember that I'm training to run a marathon.")
    resume(graph, ctx, [True])

    tomorrow = date(2026, 10, 20)
    model = Recording(responses=[AIMessage("How's training?")])
    say(graph, make_ctx(chat_model=model, day=tomorrow), "Morning!", thread_config(tomorrow))
    system = model.seen[-1][0].content
    assert "- (goal) Training to run a marathon" in system
    assert graph.get_state(thread_config(tomorrow)).values["messages"][0].content == "Morning!"  # not in messages
