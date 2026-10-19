"""Wires the nodes into a StateGraph and compiles it with a checkpointer.

    START -> router -+-> chat_agent -> END
                     +-> checkin ----> END        (one question per turn)
                     +-> summarize --> END        (stub until M5)
                     +-> extract -+-> confirm --(interrupt)--> save -> END
                     |            +-> END (nothing to save / failed)
                     +-> END (/cancel)

One invoke = one user turn. Multi-turn flows (the check-in) work because `mode` and
`checkin_step` are in the checkpointed state, so the router knows where we left off.
"""

import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from diary_agent.core.config import settings
from diary_agent.services.agent import nodes
from diary_agent.services.agent.state import DiaryContext, DiaryState

REPLY_NODES = {"chat_agent", "checkin"}  # nodes whose LLM tokens are shown to the user


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    g = StateGraph(DiaryState, context_schema=DiaryContext)
    g.add_node("router", nodes.router)
    g.add_node("chat_agent", nodes.chat_agent)
    g.add_node("checkin", nodes.checkin)
    g.add_node("summarize", nodes.summarize)
    g.add_node("extract", nodes.extract)
    g.add_node("confirm", nodes.confirm)  # returns Command(goto=...); its type hint declares the targets
    g.add_node("save", nodes.save)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        nodes.route_after_router,
        {"chat": "chat_agent", "checkin": "checkin", "extract": "extract", "summarize": "summarize", "done": END},
    )
    g.add_conditional_edges("extract", nodes.route_after_extract, ["confirm", END])
    for node in ("chat_agent", "checkin", "summarize", "save"):
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer)


def open_checkpointer(path: Path | None = None) -> SqliteSaver:
    path = path or settings.data_dir / "checkpoints.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def thread_config(day: date) -> dict:
    """thread_id = date: one conversation per day, resumable until midnight."""
    return {"configurable": {"thread_id": day.isoformat()}}


def stream_turn(
    graph: CompiledStateGraph, graph_input: dict | Command, config: dict, context: DiaryContext
) -> Iterator[tuple[str, str]]:
    """Run one turn and yield ("token", text) for reply tokens and ("status", text) for progress.

    Two stream modes at once: "messages" gives LLM tokens (tagged with the node that produced
    them), "updates" gives each node's state update as it finishes.
    """
    for mode, payload in graph.stream(graph_input, config, context=context, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = payload
            if meta.get("langgraph_node") in REPLY_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                yield "token", chunk.content
        elif payload.get("router", {}).get("route") == "extract":
            yield "status", "Writing up your entry..."


def pending_interrupt(graph: CompiledStateGraph, config: dict) -> dict | None:
    """The payload of an interrupt() the thread is paused on, if any."""
    snapshot = graph.get_state(config)
    return snapshot.interrupts[0].value if snapshot.interrupts else None
