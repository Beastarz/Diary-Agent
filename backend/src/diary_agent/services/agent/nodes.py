"""Graph nodes. A node is a plain function: (state, runtime) -> partial state update.

Only chat_agent and checkin call the chat model. The router is deterministic, and the
save path is extract (LLM) -> confirm (human) -> save (code).
"""

import re
from datetime import date, timedelta
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from diary_agent.core.prompts import CHECKIN_SYSTEM, COMPANION_SYSTEM, PROMPT_VERSION, TOOLS_SYSTEM
from diary_agent.models.entry import ExtractedEntry, JournalEntry
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services.agent.state import DiaryContext, DiaryState
from diary_agent.services.agent.tools import TOOLS
from diary_agent.services.extraction import ExtractionError, extract_entry, format_transcript

# The code picks the topic order; the model only words each question.
CHECKIN_TOPICS = (
    "how they're feeling overall today",
    "the best part of their day",
    "anything that was hard or stressful",
    "something they're grateful for",
)
MAX_TOOL_ROUNDS = 3  # per user turn; small models can keep calling tools forever
GAVE_UP = "Sorry, I couldn't find that in your journal."
# Tools are only offered for questions and requests. llama3.1 called a tool on 75% of plain
# statements ("thanks", "I'm anxious") even with a prompt rule against it (evals/tool_probe.py),
# so the code decides when tools are on the table and the model decides which one.
LOOKUP_REQUEST = re.compile(r"\?|^\s*(tell|remind|show|find|look|search|list|summari[sz]e)\b", re.IGNORECASE)
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
    system = COMPANION_SYSTEM.format(today=f"{ctx.day.isoformat()} ({ctx.day:%A})")
    if extra:
        system += "\n\n" + extra
    # The whole day's history is resent every turn. Fine for one day; trimming comes with M6.
    return [SystemMessage(system), *state["messages"]]


def calendar_hint(day: date, days_back: int = 13) -> str:
    """Code does the calendar math; the model only looks dates up. 8B models get "last Monday"
    wrong when they have to compute it (evals/tool_probe.py), but copy a listed date reliably."""
    monday = day - timedelta(days=day.weekday())
    fmt = lambda d: f"{d:%A} {d.isoformat()}"  # noqa: E731
    lines = [
        f"- today: {fmt(day)}",
        f"- yesterday: {fmt(day - timedelta(days=1))}",
        f"- this week: {monday.isoformat()} to {(monday + timedelta(days=6)).isoformat()}",
        f"- last week: {(monday - timedelta(days=7)).isoformat()} to {(monday - timedelta(days=1)).isoformat()}",
        f"- this month: {day.replace(day=1).isoformat()} to {day.isoformat()}",
        "- recent days: " + ", ".join(fmt(day - timedelta(days=i)) for i in range(2, days_back + 1)),
    ]
    return "\n".join(lines)


def _tool_rounds_this_turn(messages: list[BaseMessage]) -> int:
    rounds = 0
    for m in reversed(messages):
        if m.type == "human":
            break
        rounds += isinstance(m, AIMessage) and bool(m.tool_calls)
    return rounds


def chat_agent(state: DiaryState, runtime: Runtime[DiaryContext]) -> dict:
    """The "reason" half of the ReAct loop. If the reply has tool_calls, tools_condition sends
    them to the ToolNode, whose results come back here; otherwise the reply goes to the user."""
    ctx = runtime.context
    if not LOOKUP_REQUEST.search(state["user_input"]):
        # Statements, feelings, thanks: no tools on offer, so none can be called.
        return {"messages": [ctx.chat_model.invoke(_companion_prompt(state, ctx))]}
    prompt = _companion_prompt(state, ctx, TOOLS_SYSTEM.format(calendar=calendar_hint(ctx.day)))
    if _tool_rounds_this_turn(state["messages"]) < MAX_TOOL_ROUNDS:
        return {"messages": [ctx.chat_model.bind_tools(TOOLS).invoke(prompt)]}
    # Out of tool budget: answer with what we have. A reply that still carries tool_calls
    # would loop back to the ToolNode, so drop them.
    reply = ctx.chat_model.invoke(prompt)
    if isinstance(reply, AIMessage) and reply.tool_calls:
        reply = AIMessage(reply.content or GAVE_UP)
    return {"messages": [reply]}


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
    notice = "Saved."
    try:
        vector_repo.upsert_entry(ctx.vectors, entry)
    except Exception:  # Ollama down, etc. The entry is safe in SQLite; the index can be rebuilt.
        notice = "Saved, but not indexed for search. Run 'diary reindex' later."
    return {"draft": None, "saved_count": len(state["messages"]), "notice": notice}
