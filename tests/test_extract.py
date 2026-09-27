"""Retry logic tested with a fake extractor (a RunnableLambda), so no LLM is needed."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from diary_agent.entry import ExtractedEntry
from diary_agent.extract import ExtractionError, apply_guardrails, extract_entry, format_transcript

GOOD = ExtractedEntry(
    highlights=[], challenges=[], gratitude=[], mood_words=[], mood=6, tags=["misc"], summary="You had an ok day."
)
BAD = {"raw": AIMessage('{"mood": 11}'), "parsed": None, "parsing_error": "mood must be <= 10"}


def scripted(*results):
    calls = []

    def fn(messages):
        calls.append(messages)
        return results[len(calls) - 1]

    return RunnableLambda(fn), calls


def test_first_try_success():
    extractor, calls = scripted({"raw": AIMessage(""), "parsed": GOOD, "parsing_error": None})
    assert extract_entry(extractor, "User: ok day") == GOOD
    assert len(calls) == 1


def test_retry_feeds_error_back():
    extractor, calls = scripted(BAD, {"raw": AIMessage(""), "parsed": GOOD, "parsing_error": None})
    assert extract_entry(extractor, "User: ok day") == GOOD
    assert len(calls) == 2
    assert "mood must be <= 10" in calls[1][-1].content


def test_gives_up_after_retries():
    extractor, calls = scripted(BAD, BAD, BAD)
    with pytest.raises(ExtractionError):
        extract_entry(extractor, "User: ok day", max_retries=2)
    assert len(calls) == 3


def test_guardrail_clears_unstated_gratitude():
    entry = GOOD.model_copy(update={"gratitude": ["sister"]})
    assert apply_guardrails(entry, "User: dinner with my sister\nAssistant: I'm grateful you shared").gratitude == []
    assert apply_guardrails(entry, "User: so thankful for my sister").gratitude == ["sister"]


def test_guardrail_keeps_only_users_mood_words():
    entry = GOOD.model_copy(update={"mood_words": ["invigorated", "relieved"]})
    transcript = "User: went for a run, so relieved\nAssistant: that's invigorating!"
    assert apply_guardrails(entry, transcript).mood_words == ["relieved"]


def test_format_transcript():
    text = format_transcript([HumanMessage("hi"), AIMessage("hello")])
    assert text == "User: hi\nAssistant: hello"
