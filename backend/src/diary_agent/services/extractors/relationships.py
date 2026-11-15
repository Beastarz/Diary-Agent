"""Save-time relationships: the people the user mentioned today, and what we know about them.

transcript -> PersonMentions (LLM, one flat item per person) -> guardrails (code) -> people +
mentions tables (code). There is no human yes/no, so the guardrails carry the weight:

- The name must be in the *User's* words. This is the strongest rule: it blocks invented
  people and names the model picked up from the Assistant's replies ("How did you meet Priya?").
  A trailing 's is ignored, so "Lucas's advice" counts as naming Lucas.
- The user themself is never a person here.
- A fact must be grounded: most of its words must be words the user said.
- The same person twice in one run is merged into one mention.

Writes are idempotent per day: people are upserted (facts merged, never duplicated) and the
day's mentions are replaced as a whole, even with an empty list, so a re-run that finds nobody
also clears what an earlier run wrote for that day.

Read path: people_block() puts known people into the chat prompt when the user names them.
Plain string matching, no LLM and no embeddings, so plain chat stays as fast as before.
"""

import re
import sqlite3

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from diary_agent.core.prompts import PEOPLE_SYSTEM, RELATIONSHIPS_SYSTEM
from diary_agent.models.person import Mention, PersonMention, PersonMentions
from diary_agent.repositories import people_repository as people_repo
from diary_agent.services.extraction import field_guide, invoke_structured, user_text, words
from diary_agent.services.extractors.base import Extractor, ExtractorDeps, Report

MIN_FACT_OVERLAP = 0.8  # share of a fact's words (3+ letters) the user really said
SELF = {"i", "me", "myself", "user"}
MAX_BLOCK = 5  # people per prompt


def build(model: BaseChatModel) -> Runnable:
    return model.with_structured_output(PersonMentions, method="json_schema", include_raw=True)


def extract_candidates(runnable: Runnable, transcript: str, max_retries: int = 2) -> list[PersonMention]:
    """The model's raw list, before guardrails (the eval suite can compare the two)."""
    messages = [
        SystemMessage(f"{RELATIONSHIPS_SYSTEM}\n\nFields:\n{field_guide(PersonMention)}"),
        HumanMessage(f"Conversation:\n\n{transcript}\n\nList the people the user mentioned, if any."),
    ]
    found: PersonMentions = invoke_structured(runnable, messages, max_retries, "people list")
    return found.mentions


def _said(transcript: str) -> set[str]:
    return {re.sub(r"'s$", "", w) for w in words(user_text(transcript))}


def _grounded(fact: str, said: set[str]) -> bool:
    content = [w for w in words(fact) if len(w) > 2]
    return bool(content) and sum(w in said for w in content) / len(content) >= MIN_FACT_OVERLAP


def apply_relationship_guardrails(mentions: list[PersonMention], transcript: str) -> list[PersonMention]:
    said = _said(transcript)
    merged: dict[str, PersonMention] = {}
    for m in mentions:
        key = people_repo.normalise(m.name)
        if not key or key in SELF or not all(w in said for w in words(key)):
            continue
        facts = [f for f in m.facts if _grounded(f, said)]
        if key not in merged:
            merged[key] = m.model_copy(update={"name": m.name.strip(), "facts": facts})
            continue
        first = merged[key]
        merged[key] = first.model_copy(update={
            "facts": first.facts + [f for f in facts if f not in first.facts],
            "relation": first.relation if first.relation != "other" else m.relation,
            "what_happened": "; ".join(x for x in (first.what_happened, m.what_happened) if x),
        })
    return list(merged.values())


def extract(runnable: Runnable, transcript: str) -> list[PersonMention]:
    return apply_relationship_guardrails(extract_candidates(runnable, transcript), transcript)


def apply(items: list[PersonMention], deps: ExtractorDeps) -> Report:
    mentions, details = [], []
    for m in items:
        person, created = people_repo.upsert_person(deps.conn, m.name, m.relation, m.facts, deps.day)
        mentions.append(Mention(entry_date=deps.day, person_id=person.id, what_happened=m.what_happened,
                                evidence=m.evidence))
        details.append(f"{m.name} (new)" if created else m.name)
    people_repo.replace_mentions(deps.conn, deps.day, mentions)
    return Report("relationships", changed=len(items), details=details)


def describe(report: Report) -> str:
    if not report.changed:
        return ""
    noun = "person" if report.changed == 1 else "people"
    return f"Noted {report.changed} {noun}: {', '.join(report.details)}."


def people_block(conn: sqlite3.Connection | None, text: str) -> str:
    """System-prompt lines for the known people this message names, or "" if none.
    Matched on the normalised name as a whole word, so "Ann" never matches "planning"."""
    if conn is None:
        return ""
    named = [p for p in people_repo.list_people(conn)
             if re.search(rf"\b{re.escape(p.key)}\b", text, re.IGNORECASE)][:MAX_BLOCK]
    if not named:
        return ""
    lines = "\n".join(
        f"- {p.name} ({p.relation}): {', '.join(p.facts) or 'no details yet'}. Last seen {p.last_seen.isoformat()}."
        for p in named
    )
    return PEOPLE_SYSTEM.format(people=lines)


EXTRACTOR = Extractor(name="relationships", build=build, extract=extract, apply=apply, describe=describe)
