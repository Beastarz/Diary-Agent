"""Long-term memory about the user: facts, goals, people, preferences.

Same two layers as entry.py:
- MemoryItem: what the LLM writes (the extraction's structured output, or remember_fact's args).
- Memory: MemoryItem + what code knows (id, source date, timestamps). Stored in the LangGraph store.

Same Ollama grammar rules as entry.py: every field required, the list bounded, evidence first.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

MemoryKind = Literal["fact", "goal", "person", "preference"]


class MemoryItem(BaseModel):
    # Evidence first: quoting the user before writing the fact keeps the fact grounded, and
    # lets code check the quote is real (services/memory.py guardrails).
    evidence: str = Field(description="The user's own words this comes from, quoted exactly.")
    kind: MemoryKind = Field(
        description="goal = something they're working toward; person = someone in their life; "
        "preference = a like, dislike or routine; fact = any other lasting fact about them."
    )
    content: str = Field(
        description="One short standalone statement about the user, without 'I', e.g. "
        "'Training for a marathon in March', 'Sister Mia lives in Leeds', 'Prefers morning runs'."
    )


class MemoryCandidates(BaseModel):
    memories: list[MemoryItem] = Field(
        max_length=5,
        description="Lasting facts about the user from this conversation, one per item. Empty if none.",
    )


class Memory(MemoryItem):
    id: str
    source_date: date  # the day it was learned
    updated_at: datetime = Field(default_factory=datetime.now)
