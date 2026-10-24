"""What the graph carries. Two different things, on purpose:

- DiaryState: the conversation's *data*. Every node reads it and returns a partial update.
  The checkpointer saves it after every step, so it survives restarts. Must be serializable.
- DiaryContext: the run's *dependencies* (models, DB connection, today's date). Passed in
  fresh on every invoke via `context=`, never checkpointed. Nodes get it as `runtime.context`.
"""

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langgraph.graph import MessagesState

Route = Literal["chat", "checkin", "extract", "summarize", "done"]


def merge_pending(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """Reducer for pending_memories. Appends, because several remember_fact calls can finish in
    the same step (ToolNode runs them in parallel) and a plain key accepts one write per step.
    Writing None clears it."""
    if new is None:
        return []
    return (old or []) + new


class DiaryState(MessagesState):
    """`messages` (from MessagesState) uses the add_messages reducer: a node returning
    {"messages": [msg]} *appends*. Every other key is last-write-wins.

    TypedDict keys have no defaults, so nodes read optional keys with state.get().
    """

    user_input: str  # this turn's raw text. Commands like /save never reach `messages`.
    route: Route  # written by router, read by the conditional edge right after it
    mode: Literal["chat", "checkin"]  # persists across turns: are we mid check-in?
    checkin_step: int  # check-in questions asked so far
    draft: dict | None  # ExtractedEntry as JSON, waiting for the human's yes/no
    saved_count: int  # len(messages) at the last save; more than that = unsaved chat
    notice: str | None  # one-line status for the CLI ("Saved."); not part of the conversation
    pending_memories: Annotated[list[dict], merge_pending]  # memory proposals waiting for a yes/no


@dataclass(frozen=True)
class DiaryContext:
    chat_model: BaseChatModel
    extractor: Runnable  # build_extractor(...) output
    conn: sqlite3.Connection  # journal.db
    vectors: VectorStore  # vector_repository.open_store(): entry embeddings for search
    day: date  # the entry date, and the thread_id
    model_name: str  # recorded on saved entries
    # Long-term memory is optional: without it (or without a store on the graph) the memory
    # nodes skip. The store itself isn't here: it's passed to compile() and arrives as runtime.store.
    memory_extractor: Runnable | None = None
