"""Conversation -> ExtractedEntry, with validation-error retries.

with_structured_output(method="json_schema") makes Ollama constrain decoding to the
schema, so the JSON is almost always well-formed. Pydantic can still reject it
(e.g. mood=11), and the *content* can still be wrong - that's what evals are for.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from diary_agent.models.entry import ExtractedEntry
from diary_agent.core.llm import get_chat_model
from diary_agent.core.prompts import EXTRACT_SYSTEM


class ExtractionError(Exception):
    pass


# Rules the schema states but the model still breaks (NOTES.md, M2). Enforcing them in code
# is cheap, deterministic and testable, and unlike prompt tweaks it can't regress other cases.
GRATITUDE_WORDS = ("grateful", "thankful", "thank", "appreciat", "blessed")


def apply_guardrails(entry: ExtractedEntry, transcript: str) -> ExtractedEntry:
    user_text = " ".join(line for line in transcript.splitlines() if line.startswith("User:")).lower()
    updates = {}
    # Gratitude must be explicit.
    if entry.gratitude and not any(w in user_text for w in GRATITUDE_WORDS):
        updates["gratitude"] = []
    # Mood words must be the user's own (stem match, so "relief" keeps "relieved").
    # Stops the assistant's words ("invigorating") from becoming the user's feelings.
    kept = [w for w in entry.mood_words if w.lower()[:5] in user_text]
    if kept != entry.mood_words:
        updates["mood_words"] = kept
    return entry.model_copy(update=updates) if updates else entry


def get_extraction_model() -> BaseChatModel:
    # temperature 0: we want the most likely reading, not creativity.
    # max_tokens: safety net against runaway generation (an entry is ~150 tokens).
    return get_chat_model(temperature=0, max_tokens=512)


def build_extractor(model: BaseChatModel) -> Runnable:
    """Returns a runnable that yields {"raw": AIMessage, "parsed": ExtractedEntry | None, "parsing_error": ...}."""
    return model.with_structured_output(ExtractedEntry, method="json_schema", include_raw=True)


def field_guide(schema: type[BaseModel] = ExtractedEntry) -> str:
    """Schema descriptions -> prompt text, so the schema stays the single source of truth."""
    return "\n".join(f"- {name}: {f.description}" for name, f in schema.model_fields.items())


def format_transcript(history: list[BaseMessage]) -> str:
    """Only what was *said* about today. Left out, so old days don't leak into today's entry:
    - tool results (they're past entries) and tool-call messages (usually empty)
    - the assistant's answer after a tool result, until the user speaks again. It retells past
      days ("Last Saturday you ran a 10k"), and extraction copied those into today's highlights
      every time (evals: past_day_leak). A "today only" prompt rule was tried instead: it fixed the
      leak but pulled the assistant's words back in ("enjoyed the invigorating feeling"), so it
      scored no better (71% vs 72% all-checks, 2 x 3 runs each). The code knows which replies
      came from a lookup; the model has to guess.
    """
    speaker = {"human": "User", "ai": "Assistant"}
    lines, after_lookup = [], False
    for m in history:
        if m.type == "human":
            after_lookup = False
        elif m.type == "tool":
            after_lookup = True
        if m.type in speaker and m.content and not (m.type == "ai" and after_lookup):
            lines.append(f"{speaker[m.type]}: {m.content}")
    return "\n".join(lines)


def extract_entry(extractor: Runnable, transcript: str, max_retries: int = 2) -> ExtractedEntry:
    messages: list[BaseMessage] = [
        SystemMessage(f"{EXTRACT_SYSTEM}\n\nFields:\n{field_guide()}"),
        HumanMessage(f"Conversation:\n\n{transcript}\n\nExtract the diary entry."),
    ]
    return apply_guardrails(invoke_structured(extractor, messages, max_retries, "entry"), transcript)


def invoke_structured(runnable: Runnable, messages: list[BaseMessage], max_retries: int, what: str):
    """Calls a with_structured_output(include_raw=True) runnable until it parses. Also used by summaries."""
    messages = list(messages)
    for _ in range(max_retries + 1):
        result = runnable.invoke(messages)
        if result["parsed"] is not None:
            return result["parsed"]
        # Show the model its own output and what was wrong, then ask again.
        raw = result["raw"].content if result.get("raw") else ""
        messages += [
            AIMessage(raw),
            HumanMessage(f"That output was invalid: {result['parsing_error']}. Return corrected JSON only."),
        ]
    raise ExtractionError(f"Could not get a valid {what} after {max_retries + 1} attempts")
