"""Wires the nodes into a StateGraph and compiles it with a checkpointer.

                                        START
                                          |
                                          v
                                   +-------------+
                                   |   router    |  rule-based: slash commands, the "check in"
                                   +-------------+  keyword, and the checkpointed `mode`
                                          |
          +---------------+---------------+---------------+---------------+
          | chat          | checkin       | summarize     | extract       | done (/cancel)
          v               v               v               v               v
    +------------+  +-----------+  +-------------+  +-----------+        END
    | chat_agent |  |  checkin  |  |  summarize  |  |  extract  |
    +------------+  +-----------+  +-------------+  +-----------+
      |    |   ^          |               |            |      |
      |    |   |          v               v      draft |      | nothing to save
      |    |   |         END             END           |      | or extraction failed
      |    |   |                                       v      v
      |    v   |                                  +-------+  END
      |  +--------+                               | save  |  writes the entry: no yes/no, a later
      |  | tools  |  ToolNode runs the calls,     +-------+  save the same day replaces it
      |  +--------+  results go back up               |
      |      |                                        v
      | no   | remember_fact                 +----------------+  services/extractors: memory,
      | tool | proposed a memory             | run_extractors |  people, ... each mines the
      | calls|                               +----------------+  transcript and writes its own
      v      v                                        |          records. Failures go in the notice.
     END  +----------------+                          v
          | confirm_memory |  interrupt():           END
          +----------------+  "Remember this?" per proposal, then writes the store,
                              then -> chat_agent to answer the remember_fact call.

One invoke = one user turn. Multi-turn flows (the check-in) work because `mode` and
`checkin_step` are in the checkpointed state, so the router knows where we left off.
"""

import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.types import Command

from diary_agent.core.config import settings
from diary_agent.services.agent import nodes
from diary_agent.services.agent.state import DiaryContext, DiaryState
from diary_agent.services.agent.tools import TOOLS

REPLY_NODES = {"chat_agent", "checkin"}  # nodes whose LLM tokens are shown to the user


def build_graph(checkpointer: BaseCheckpointSaver | None = None, store: BaseStore | None = None) -> CompiledStateGraph:
    """checkpointer = short-term memory (one thread per day); store = long-term memory (shared by all threads)."""
    g = StateGraph(DiaryState, context_schema=DiaryContext)
    g.add_node("router", nodes.router)
    g.add_node("chat_agent", nodes.chat_agent)
    g.add_node("tools", ToolNode(TOOLS))  # handle_tool_errors=True by default: exceptions become ToolMessages
    g.add_node("checkin", nodes.checkin)
    g.add_node("summarize", nodes.summarize)
    g.add_node("extract", nodes.extract)
    g.add_node("save", nodes.save)
    g.add_node("run_extractors", nodes.run_extractors)
    g.add_node("confirm_memory", nodes.confirm_memory)  # returns Command(goto=...); its type hint declares the targets

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        nodes.route_after_router,
        {"chat": "chat_agent", "checkin": "checkin", "extract": "extract", "summarize": "summarize", "done": END},
    )
    g.add_conditional_edges("chat_agent", tools_condition, ["tools", END])
    g.add_conditional_edges("tools", nodes.route_after_tools, ["confirm_memory", "chat_agent"])
    g.add_conditional_edges("extract", nodes.route_after_extract, ["save", END])
    g.add_edge("save", "run_extractors")
    g.add_edge("run_extractors", END)
    for node in ("checkin", "summarize"):
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer, store=store)


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
            # Streaming models send AIMessageChunks; a message built in code (or by a
            # non-streaming model) arrives whole. AIMessageChunk is a subclass of AIMessage.
            if meta.get("langgraph_node") in REPLY_NODES and isinstance(chunk, AIMessage) and chunk.content:
                yield "token", chunk.content
        elif payload.get("router", {}).get("route") == "extract":
            yield "status", "Writing up your entry..."
        elif "save" in payload and context.extractors:
            yield "status", "Noting people and things to remember..."
        elif any(c["name"] != "remember_fact" for m in (payload.get("chat_agent") or {}).get("messages", [])
                 for c in getattr(m, "tool_calls", None) or []):
            yield "status", "Searching your journal..."


def pending_interrupt(graph: CompiledStateGraph, config: dict) -> dict | None:
    """The payload of an interrupt() the thread is paused on, if any."""
    snapshot = graph.get_state(config)
    return snapshot.interrupts[0].value if snapshot.interrupts else None
