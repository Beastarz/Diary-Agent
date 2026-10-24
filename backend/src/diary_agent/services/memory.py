"""Long-term memory: what gets written (write policy) and what gets read back (injection).

Write path: transcript -> candidates (LLM) -> guardrails (code) -> proposals (code, dedupe
against the store) -> human yes/no (graph interrupt) -> store. Two sources feed it:
- update_memory: after an entry is saved, the day's conversation is mined for lasting facts.
- remember_fact: the user explicitly says "remember that ...".
Every write is confirmed by the user. A wrong memory is worse than a wrong entry: it gets
injected into every future conversation ("memory pollution").

Read path: memory_block() turns the store into a few lines for the system prompt.
"""

import re
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.store.base import BaseStore

from diary_agent.core.prompts import EXTRACT_MEMORY_SYSTEM, MEMORY_SYSTEM
from diary_agent.models.memory import Memory, MemoryCandidates, MemoryItem
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.services.extraction import field_guide, invoke_structured

# Cosine similarity with the nearest stored memory (nomic-embed-text, evals/memory_probe.py):
# rewordings of one fact 0.83-0.94, updates 0.55-0.75, different facts 0.38-0.58. Embeddings
# measure topic, not "same fact": "Lives in London" -> "Moved to Leeds" (0.55) looks like
# "Sister Mia" vs "Brother Tom" (0.58). So the costly error to avoid is an *update* that
# overwrites a different fact; RELATED sits above every different pair seen. Missed updates
# become a second memory instead, which the user can 'diary memory forget'.
DUPLICATE_SCORE = 0.90  # same fact: skip
RELATED_SCORE = 0.70  # same kind and this close: propose replacing it (the user sees old -> new)
MAX_CONTENT_CHARS = 120
MIN_EVIDENCE_OVERLAP = 0.8  # share of the quote's words the user really said
# Words that mark a one-day state or event, not a lasting fact.
FIRST_PERSON = {"i", "i'm", "i've", "i'd", "i'll", "my", "me", "mine", "myself", "we", "our", "us"}
THIRD_PERSON = {"he", "she", "they", "he's", "she's", "they're", "his", "her", "their", "him", "them"}
TRANSIENT = re.compile(r"\b(today|tonight|yesterday|tomorrow|this (morning|afternoon|evening)|right now)\b", re.I)

# Read path: a small profile goes into the prompt whole; past this, goals + the best matches.
MAX_INJECTED = 15
RELEVANT_K = 5
RELEVANT_SCORE = 0.5


def build_memory_extractor(model: BaseChatModel) -> Runnable:
    return model.with_structured_output(MemoryCandidates, method="json_schema", include_raw=True)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _user_text(transcript: str) -> str:
    return " ".join(line.removeprefix("User:") for line in transcript.splitlines() if line.startswith("User:"))


def apply_memory_guardrails(items: list[MemoryItem], transcript: str) -> list[MemoryItem]:
    """Deterministic write policy. Like apply_guardrails for entries: rules the prompt states
    but an 8B model still breaks, enforced where they can't regress."""
    said = set(_words(_user_text(transcript)))
    kept, seen = [], set()
    for item in items:
        quote = _words(item.evidence)
        content = item.content.strip()
        if not quote or not content or len(content) > MAX_CONTENT_CHARS:
            continue
        # The quote must be the user's words, not the assistant's and not invented.
        if sum(w in said for w in quote) / len(quote) < MIN_EVIDENCE_OVERLAP:
            continue
        if TRANSIENT.search(content):  # "Feeling tired today" is an entry, not a memory
            continue
        # Wrong subject: from "my sister Mia... She just moved to Leeds" llama3.1 saved
        # "Lives in Leeds" as a fact about the *user* (evals/memory_probe.py). A quote about
        # he/she/they with no I/my/me is about someone else, so only a `person` memory may use it.
        if item.kind != "person" and not FIRST_PERSON & set(quote) and THIRD_PERSON & set(quote):
            continue
        key = " ".join(_words(content))
        if key in seen:
            continue
        seen.add(key)
        kept.append(item.model_copy(update={"content": content}))
    return kept


def extract_candidates(extractor: Runnable, transcript: str, max_retries: int = 2) -> list[MemoryItem]:
    """The model's raw list, before guardrails (evals/memory_probe.py compares the two)."""
    messages = [
        SystemMessage(f"{EXTRACT_MEMORY_SYSTEM}\n\nFields:\n{field_guide(MemoryItem)}"),
        HumanMessage(f"Conversation:\n\n{transcript}\n\nList the lasting facts about the user, if any."),
    ]
    found: MemoryCandidates = invoke_structured(extractor, messages, max_retries, "memory list")
    return found.memories


def extract_memories(extractor: Runnable, transcript: str, max_retries: int = 2) -> list[MemoryItem]:
    return apply_memory_guardrails(extract_candidates(extractor, transcript, max_retries), transcript)


def propose(
    store: BaseStore, items: list[MemoryItem], day: date, tool_call_id: str | None = None
) -> list[dict]:
    """Candidates -> proposals for the human. Plain JSON, because they wait in checkpointed state.

    {"action": "add" | "update", "memory": Memory, "replaces": old content | None,
     "tool_call_id": the remember_fact call this answers | None}
    """
    proposals = []
    for item in items:
        try:
            hits = memory_repo.search(store, item.content, k=1)
        except Exception:  # embeddings down: can't dedupe, so propose it as new
            hits = []
        nearest, score = hits[0] if hits else (None, 0.0)
        if nearest and score >= DUPLICATE_SCORE:
            continue
        if nearest and score >= RELATED_SCORE and nearest.kind == item.kind:
            memory = Memory(**item.model_dump(), id=nearest.id, source_date=day)
            proposals.append({"action": "update", "memory": memory.model_dump(mode="json"),
                              "replaces": nearest.content, "tool_call_id": tool_call_id})
            continue
        memory = Memory(**item.model_dump(), id=memory_repo.new_id(), source_date=day)
        proposals.append({"action": "add", "memory": memory.model_dump(mode="json"),
                          "replaces": None, "tool_call_id": tool_call_id})
    return proposals


def apply_proposals(store: BaseStore, proposals: list[dict]) -> None:
    for p in proposals:
        memory_repo.put_memory(store, Memory.model_validate(p["memory"]))  # same id = update in place


def select_for_prompt(store: BaseStore, query: str) -> list[Memory]:
    memories = memory_repo.list_memories(store)
    if len(memories) <= MAX_INJECTED:
        return memories
    picked = [m for m in memories if m.kind == "goal"][:RELEVANT_K]
    try:
        hits = memory_repo.search(store, query, k=RELEVANT_K)
    except Exception:  # embeddings down: fall back to the most recent ones
        hits = [(m, 1.0) for m in memories[:RELEVANT_K]]
    ids = {m.id for m in picked}
    picked += [m for m, score in hits if score >= RELEVANT_SCORE and m.id not in ids]
    return picked


def memory_block(store: BaseStore | None, query: str) -> str:
    """System-prompt text listing what we know about the user, or "" if nothing."""
    if store is None:
        return ""
    memories = select_for_prompt(store, query)
    if not memories:
        return ""
    lines = "\n".join(f"- ({m.kind}) {m.content}" for m in memories)
    return MEMORY_SYSTEM.format(memories=lines)
