"""Weekly / monthly summary schemas.

Same two layers as entry.py:
- PeriodSummary: only what the LLM must write (the reduce step's structured output).
- SummaryRecord: PeriodSummary + facts code already knows (range, entry count, mood stats).
  The mood numbers are computed, never asked for: an 8B model averaging 30 numbers is a coin flip.

Same Ollama grammar rules as entry.py: every field required, every list bounded, evidence first.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Period = Literal["week", "month"]


class PeriodSummary(BaseModel):
    themes: list[str] = Field(
        max_length=5,
        description="Recurring themes: topics, people or activities that came up on 2 or more days, "
        "as short phrases. Empty if nothing recurred.",
    )
    wins: list[str] = Field(
        max_length=5,
        description="Good things that happened, achievements or progress, as short phrases naming the event. "
        "Only things the digests mention. A mood ('good mood') is not a win.",
    )
    challenges: list[str] = Field(
        max_length=5,
        description="Difficulties or stressors from the period, as short phrases. Only things the digests mention.",
    )
    mood_trend: str = Field(
        description="1-2 sentences on how mood moved over the period and what seemed to drive the high "
        "and low days. Use the computed mood numbers given; do not recalculate them.",
    )
    overview: str = Field(
        description="2-4 sentence overview of the period in second person ('You ...'), "
        "based only on the digests.",
    )


class SummaryRecord(PeriodSummary):
    period: Period
    start_date: date
    end_date: date
    entry_count: int
    mood_avg: float | None  # None when no entry in the range had a mood
    model: str
    prompt_version: str
    created_at: datetime = Field(default_factory=datetime.now)
