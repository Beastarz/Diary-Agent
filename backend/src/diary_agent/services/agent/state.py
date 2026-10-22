"""What the graph carries. Two different things, on purpose:

- DiaryState: the conversation's *data*. Every node reads it and returns a partial update.
  The checkpointer saves it after every step, so it survives restarts. Must be serializable.
- DiaryContext: the run's *dependencies* (models, DB connection, today's date). Passed in
  fresh on every invoke via `context=`, never checkpointed. Nodes get it as `runtime.context`.
"""

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langgraph.graph import MessagesState

Route = Literal["chat", "checkin", "extract", "summarize", "done"]


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


@dataclass(frozen=True)
class DiaryContext:
    chat_model: BaseChatModel
    extractor: Runnable  # build_extractor(...) output
    conn: sqlite3.Connection  # journal.db
    vectors: VectorStore  # vector_repository.open_store(): entry embeddings for search
    day: date  # the entry date, and the thread_id
    model_name: str  # recorded on saved entries
