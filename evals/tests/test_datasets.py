"""Labels are data, and data has typos. A misspelled route or a gold date that isn't in the
journal would quietly score as a model failure; these catch that before any model runs."""

from datetime import date

from suites import load_jsonl

ROUTES = {"chat", "checkin", "extract", "summarize", "done"}
EXTRACTION_KEYS = {"id", "transcript", "messages", "mood", "null_ok", "mentions", "placement", "not_in", "empty", "forbidden",
                   "tags_any", "tags_not", "mood_words_any"}
LIST_FIELDS = {"highlights", "challenges", "gratitude", "mood_words", "tags"}


def unique_ids(rows):
    ids = [r["id"] for r in rows]
    return len(ids) == len(set(ids))


def test_routing_labels():
    rows = load_jsonl("routing.jsonl")
    assert unique_ids(rows)
    for r in rows:
        assert r["route"] in ROUTES, r["id"]
        assert r.get("tools", "none") in {"none", "read", "all"}, r["id"]
        assert "tools" not in r or r["route"] == "chat", f"{r['id']}: tools only apply to the chat route"


def test_retrieval_gold_days_exist_in_the_journal():
    days = {date.fromisoformat(r["date"]) for r in load_jsonl("retrieval_journal.jsonl")}
    questions = load_jsonl("retrieval_questions.jsonl")
    assert unique_ids(questions)
    for q in questions:
        assert all(date.fromisoformat(d) in days for d in q["gold"]), q["id"]


def test_extraction_cases_use_known_keys_and_fields():
    rows = load_jsonl("extraction.jsonl")
    assert unique_ids(rows)
    for r in rows:
        assert set(r) <= EXTRACTION_KEYS, f"{r['id']}: unknown keys {set(r) - EXTRACTION_KEYS}"
        assert ("transcript" in r) != ("messages" in r), f"{r['id']}: give a transcript or messages, not both"
        assert all(t["role"] in {"user", "assistant", "tool_call", "tool"} for t in r.get("messages", [])), r["id"]
        lo, hi = r["mood"]
        assert 1 <= lo <= hi <= 10, r["id"]
        for key in ("placement", "not_in"):
            assert set(r.get(key, {})) <= LIST_FIELDS, r["id"]
        assert set(r.get("empty", [])) <= LIST_FIELDS, r["id"]


def test_message_cases_go_through_format_transcript():
    from suites.extraction import to_transcript

    case = next(r for r in load_jsonl("extraction.jsonl") if r["id"] == "past_day_leak")
    transcript = to_transcript(case)
    assert "10k" not in transcript  # the lookup and its answer are dropped
    assert transcript.endswith("made pasta.")


def test_routing_suite_runs_without_a_model():
    from suites import routing

    out = routing.run(runs=1)
    assert out["metrics"]["n_cases"] == len(load_jsonl("routing.jsonl"))
    assert 0 <= out["metrics"]["route_accuracy"] <= 1


def test_tools_cases_reference_real_tools_and_days():
    from diary_agent.services.agent.tools import TOOLS

    names = {t.name for t in TOOLS}
    days = {date.fromisoformat(r["date"]) for r in load_jsonl("tools_journal.jsonl")}
    rows = load_jsonl("tools.jsonl")
    assert unique_ids(rows)
    for r in rows:
        assert set(r["tools"]) <= names, r["id"]
        for key in ("cover", "cover_any", "exclude"):
            for d in r.get(key, []):
                date.fromisoformat(d)  # valid date
        assert "answer" not in r or r["tools"], f"{r['id']}: answer checks need a lookup"
        assert all(date.fromisoformat(d) in days for d in r.get("cover", [])), r["id"]


def test_language_suite_flags_only_cjk():
    from suites import language

    out = language.run_on({"tools": ["Hello!", "好的"], "extraction": ["You ran."]})
    assert out["metrics"] == {"no_cjk": 2 / 3, "n_texts": 3}
    assert len(out["failures"]) == 1


def test_chat_cases_use_known_checks():
    from judge import CHECKS, RUBRICS

    assert set(RUBRICS) == set(CHECKS)
    rows = load_jsonl("chat.jsonl")
    assert unique_ids(rows)
    for r in rows:
        assert r["turns"] and set(r["checks"]) <= set(CHECKS), r["id"]
    ids = {r["id"]: r for r in rows}
    for p in load_jsonl("judge_planted.jsonl"):
        assert p["case"] in ids, p["case"]
        assert set(p["fails"]) <= set(ids[p["case"]]["checks"]), f"{p['case']}: planted fail on a check not graded"


def test_judge_sees_what_the_grader_sees():
    from judge import view

    text = view("User: hi", "", "Hello!")
    assert "(nothing looked up)" in text and text.endswith("Hello!")


def test_gold_file_is_graded_with_known_values():
    from judge_calibrate import CHECKS, GOLD, GRADES, read_gold

    if not GOLD.exists():
        return  # not sampled yet
    for r in read_gold():
        for c in CHECKS:
            assert r[c].strip().lower() in {"", "n/a", *GRADES}, f"row {r['row']} {c}: {r[c]!r}"
