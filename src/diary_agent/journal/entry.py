"""Journal entry schemas.

Two layers on purpose:
- ExtractedEntry: only what the LLM must infer from the conversation.
- JournalEntry: ExtractedEntry + facts code already knows (date, transcript, model).
  Never ask a model for something you can compute.

Ollama compiles this schema into a decoding grammar. Lessons learned the hard way (NOTES.md, M2):
- Every field is REQUIRED (no defaults). The grammar only forces required fields, so optional
  ones were silently skipped. "Nothing to say" is expressed as [] or null instead.
- Every list has max_length. An unbounded list let the model repeat tags until it hit the token cap.
- Field ORDER matters: evidence fields first, the mood judgement after the facts are written down.
- Descriptions do NOT reach the model via the grammar; extract.field_guide() puts them in the prompt.
"""

from datetime import date, datetime

from pydantic import BaseModel, Field


class ExtractedEntry(BaseModel):
    highlights: list[str] = Field(
        max_length=5,
        description="Positive or meaningful things that happened, as short phrases. "
        "Only things the user actually said. Negative events go in challenges, not here.",
    )
    challenges: list[str] = Field(
        max_length=5,
        description="Difficulties, stressors or frustrations, as short phrases. Only things the user actually said.",
    )
    gratitude: list[str] = Field(
        max_length=5,
        description="Things the user explicitly said they were grateful or thankful for. "
        "Empty unless they used words like 'grateful', 'thankful' or 'appreciate'.",
    )
    mood_words: list[str] = Field(
        max_length=3,
        description="1-3 single words for feelings the User expressed about themselves, in their own words "
        "(e.g. 'tired', 'relieved'). Never use words the Assistant said. "
        "Empty if the User never described their feelings.",
    )
    mood: int | None = Field(
        ge=1,
        le=10,
        description="Overall mood for the day: 1-3 bad, 4-6 mixed or neutral, 7-10 good. "
        "Infer it from the events and feelings the user described. "
        "Use null only if the user shared no events or feelings that suggest a mood.",
    )
    tags: list[str] = Field(
        max_length=5,
        description="2-5 lowercase one-word topic tags, e.g. 'work', 'family', 'health', 'food'.",
    )
    summary: str = Field(
        description="1-2 sentence summary of the day in second person ('You ...'), "
        "based only on what the User said. Do not add the Assistant's interpretations.",
    )


class JournalEntry(ExtractedEntry):
    entry_date: date
    transcript: str
    model: str
    prompt_version: str
    created_at: datetime = Field(default_factory=datetime.now)
