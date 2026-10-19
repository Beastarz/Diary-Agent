"""Graph nodes. A node is a plain function: (state, runtime) -> partial state update.

Only chat_agent and checkin call the chat model. The router is deterministic, and the
save path is extract (LLM) -> confirm (human) -> save (code).
"""

import re
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from diary_agent.core.prompts import CHECKIN_SYSTEM, COMPANION_SYSTEM, PROMPT_VERSION
from diary_agent.models.entry import ExtractedEntry, JournalEntry
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.services.agent.state import DiaryContext, DiaryState
from diary_agent.services.extraction import ExtractionError, extract_entry, format_transcript

# The code picks the topic order; the model only words each question.
CHECKIN_TOPICS = (
    "how they're feeling overall today",
    "the best part of their day",
    "anything that was hard or stressful",
    "something they're grateful for",
)
CHECKIN_PHRASE = re.compile(r"\bcheck[- ]?in\b", re.IGNORECASE)


def router(state: DiaryState) -> dict:
    """Rule-based: slash commands plus one keyword. Cheap, instant and testable.
    An LLM fallback for fuzzy intents can come later, once M7 routing evals can measure it."""
    text = state["user_input"].strip()
    cmd = text.lower()
    update: dict = {"notice": None}  # clear last turn's notice
    if cmd == "/save":
        return {**update, "route": "extract", "mode": "chat"}
    if cmd == "/summary":
        return {**update, "route": "summarize"}
    if cmd == "/cancel":
        return {**update, "route": "done", "mode": "chat", "notice": "Check-in cancelled."}

    in_checkin = state.get("mode") == "checkin"
    if cmd == "/checkin":
        text = "Let's do a check-in."  # the model needs a user turn to respond to
    if not in_checkin and (cmd == "/checkin" or CHECKIN_PHRASE.search(text)):
        return {**update, "messages": [HumanMessage(text)], "route": "checkin", "mode": "checkin", "checkin_step": 0}

    update["messages"] = [HumanMessage(text)]
    if in_checkin and state.get("checkin_step", 0) >= len(CHECKIN_TOPICS):
        return {**update, "route": "extract", "mode": "chat"}  # last answer is in: write it up
    return {**update, "route": "checkin" if in_checkin else "chat"}


def route_after_router(state: DiaryState) -> str:
    """The conditional edge: just reads what the router decided."""
    return state["route"]


def _companion_prompt(state: DiaryState, ctx: DiaryContext, extra: str = "") -> list[BaseMessage]:
    system = COMPANION_SYSTEM.format(today=ctx.day.isoformat())
    if extra:
        system += "\n\n" + extra
    # The whole day's history is resent every turn. Fine for one day; trimming comes with M6.
    return [SystemMessage(system), *state["messages"]]


def chat_agent(state: DiaryState, runtime: Runtime[DiaryContext]) -> dict:
    ctx = runtime.context
    return {"messages": [ctx.chat_model.invoke(_companion_prompt(state, ctx))]}


def checkin(state: DiaryState, runtime: Runtime[DiaryContext]) -> dict:
    ctx = runtime.context
    step = state.get("checkin_step", 0)
    hint = CHECKIN_SYSTEM.format(step=step + 1, total=len(CHECKIN_TOPICS), topic=CHECKIN_TOPICS[step])
    reply = ctx.chat_model.invoke(_companion_prompt(state, ctx, hint))
    return {"messages": [reply], "checkin_step": step + 1}


def summarize(state: DiaryState) -> dict:
    return {"notice": "Summaries arrive in M5."}


def _transcript(state: DiaryState) -> str:
    return format_transcript(state["messages"])


def extract(state: DiaryState, runtime: Runtime[DiaryContext]) -> dict:
    if not any(m.type == "human" for m in state["messages"]):
        return {"draft": None, "notice": "Nothing to save yet - tell me about your day first."}
    try:
        entry = extract_entry(runtime.context.extractor, _transcript(state))
    except ExtractionError as e:
        return {"draft": None, "notice": str(e)}
    # Stored as plain JSON: checkpoints must be serializable.
    return {"draft": entry.model_dump(mode="json")}


def route_after_extract(state: DiaryState) -> str:
    return "confirm" if state.get("draft") else END


def confirm(state: DiaryState, runtime: Runtime[DiaryContext]) -> Command[Literal["save", "__end__"]]:
    """Human-in-the-loop. interrupt() pauses the graph and checkpoints it. The CLI shows the
    payload, then resumes with Command(resume=True/False), which becomes interrupt()'s return value.

    On resume the node runs again *from the top*, so everything before interrupt() must be
    safe to repeat. A DB read is; the write lives in the next node for that reason.
    """
    exists = entry_repo.get_entry(runtime.context.conn, runtime.context.day) is not None
    approved = interrupt({"draft": state["draft"], "exists": exists})
    if approved:
        return Command(goto="save")
    return Command(goto=END, update={"draft": None, "notice": "Not saved."})


def save(state: DiaryState, runtime: Runtime[DiaryContext]) -> dict:
    ctx = runtime.context
    entry = JournalEntry(
        **ExtractedEntry.model_validate(state["draft"]).model_dump(),
        entry_date=ctx.day,
        transcript=_transcript(state),
        model=ctx.model_name,
        prompt_version=PROMPT_VERSION,
    )
    entry_repo.save_entry(ctx.conn, entry)
    return {"draft": None, "saved_count": len(state["messages"]), "notice": "Saved."}
