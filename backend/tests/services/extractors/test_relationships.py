"""The relationships extractor: guardrails, idempotent apply, the notice, and the prompt block."""

from datetime import date

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from diary_agent.models.person import PersonMention, PersonMentions
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import people_repository as people_repo
from diary_agent.services.extractors import relationships as rel
from diary_agent.services.extractors.base import ExtractorDeps

DAY = date(2026, 11, 15)
TRANSCRIPT = (
    "User: I met a new friend who is good at math. Lucas's advice helped.\n"
    "Assistant: How did you meet Priya?\n"
    "User: Also saw my mum."
)


def mention(name, relation="friend", facts=(), what="met", evidence="I met Lucas"):
    return PersonMention(evidence=evidence, name=name, relation=relation, facts=list(facts), what_happened=what)


def test_guardrails_keep_a_name_the_user_wrote_in_another_form():
    kept = rel.apply_relationship_guardrails([mention("lucas"), mention("Lucas's")], TRANSCRIPT)
    assert [m.name for m in kept] == ["lucas"]  # merged into one


def test_guardrails_drop_names_only_the_assistant_said_or_nobody_said():
    kept = rel.apply_relationship_guardrails([mention("Priya"), mention("Tom")], TRANSCRIPT)
    assert kept == []


def test_guardrails_drop_the_user_and_ungrounded_facts():
    kept = rel.apply_relationship_guardrails(
        [mention("me"), mention("Lucas", facts=["good at math", "plays the violin"])], TRANSCRIPT)
    assert [m.name for m in kept] == ["Lucas"]
    assert kept[0].facts == ["good at math"]


def test_guardrails_drop_everything_when_the_user_named_nobody():
    assert rel.apply_relationship_guardrails([mention("Lucas")], "User: quiet day alone.\nAssistant: Nice.") == []


def test_apply_upserts_people_and_replaces_the_days_mentions(tmp_path):
    conn = entry_repo.connect(tmp_path / "journal.db")
    deps = ExtractorDeps(conn=conn, store=None, day=DAY, model_name="fake")
    report = rel.apply([mention("Lucas", facts=["good at math"]), mention("my mum", "family")], deps)
    assert rel.describe(report) == "Noted 2 people: Lucas (new), my mum (new)."
    report = rel.apply([mention("Lucas", what="chess")], deps)
    assert rel.describe(report) == "Noted 1 person: Lucas."
    assert [(p.name, m.what_happened) for p, m in people_repo.mentions_on(conn, DAY)] == [("Lucas", "chess")]
    assert people_repo.get_person(conn, "mum") is not None  # the person stays; only the day's mention went


def test_describe_is_empty_when_nothing_changed(tmp_path):
    conn = entry_repo.connect(tmp_path / "journal.db")
    report = rel.apply([], ExtractorDeps(conn=conn, store=None, day=DAY, model_name="fake"))
    assert rel.describe(report) == "" and report.changed == 0


def test_extract_runs_the_runnable_then_the_guardrails():
    parsed = PersonMentions(mentions=[mention("Lucas"), mention("Priya")])
    runnable = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": parsed, "parsing_error": None})
    assert [m.name for m in rel.extract(runnable, TRANSCRIPT)] == ["Lucas"]


def test_people_block_matches_whole_words_only(tmp_path):
    conn = entry_repo.connect(tmp_path / "journal.db")
    people_repo.upsert_person(conn, "Ann", "friend", ["likes hiking"], DAY)
    people_repo.upsert_person(conn, "Lucas", "friend", [], DAY)
    assert rel.people_block(conn, "I'm planning a trip") == ""
    assert rel.people_block(None, "saw ann") == ""
    block = rel.people_block(conn, "saw ann and lucas")
    assert "- Ann (friend): likes hiking. Last seen 2026-11-15." in block
    assert "- Lucas (friend): no details yet. Last seen 2026-11-15." in block
