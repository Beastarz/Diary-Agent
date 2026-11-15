"""People the user mentions: who they are, and what happened with them on a given day.

Same two layers as entry.py and memory.py:
- PersonMention / PersonMentions: what the LLM writes, one flat item per person mentioned.
- Person / Mention: what is stored. Code builds a Person from the mention's name, relation and
  facts, and a Mention from the rest, so the model never has to fill a nested schema.

Same Ollama grammar rules: every field required, lists bounded, evidence first.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Relation = Literal["family", "partner", "friend", "colleague", "acquaintance", "other"]


class PersonMention(BaseModel):
    evidence: str = Field(description="The User's own words that mention this person, quoted exactly.")
    name: str = Field(
        description="How the User refers to the person: a name ('Lucas'), or a role if no name is given "
        "('my boss', 'Mum'). Never a name only the Assistant used."
    )
    relation: Relation = Field(description="family, partner, friend, colleague, acquaintance, or other if unclear.")
    facts: list[str] = Field(
        max_length=3,
        description="Up to 3 lasting things the User said about this person, as short phrases "
        "(e.g. 'good at math', 'lives in Leeds'). Empty if none.",
    )
    what_happened: str = Field(description="One short sentence: what happened with this person today.")


class PersonMentions(BaseModel):
    mentions: list[PersonMention] = Field(
        max_length=5,
        description="One item per person the User mentioned. Empty if they mentioned nobody.",
    )


class Person(BaseModel):
    id: str
    key: str  # people_repository.normalise(name): "my mum" and "Mum" are one person
    name: str  # the first display form seen
    relation: Relation
    facts: list[str]
    first_seen: date
    last_seen: date
    updated_at: datetime = Field(default_factory=datetime.now)


class Mention(BaseModel):
    entry_date: date
    person_id: str
    what_happened: str
    evidence: str
