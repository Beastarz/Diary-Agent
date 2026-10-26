"""Labels are data, and data has typos. A misspelled route or a gold date that isn't in the
journal would quietly score as a model failure; these catch that before any model runs."""

from datetime import date

from suites import load_jsonl

ROUTES = {"chat", "checkin", "extract", "summarize", "done"}
EXTRACTION_KEYS = {"id", "transcript", "mood", "null_ok", "mentions", "placement", "not_in", "empty", "forbidden",
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
        lo, hi = r["mood"]
        assert 1 <= lo <= hi <= 10, r["id"]
        for key in ("placement", "not_in"):
            assert set(r.get(key, {})) <= LIST_FIELDS, r["id"]
        assert set(r.get("empty", [])) <= LIST_FIELDS, r["id"]


def test_routing_suite_runs_without_a_model():
    from suites import routing

    out = routing.run(runs=1)
    assert out["metrics"]["n_cases"] == len(load_jsonl("routing.jsonl"))
    assert 0 <= out["metrics"]["route_accuracy"] <= 1
