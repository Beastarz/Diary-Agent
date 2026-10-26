"""Extraction suite: transcripts with gold labels -> extract_entry(), the real save-time path
(structured output, retries, guardrails). Summaries and search both read these fields, so this
is the suite that matters most. Replaces M2's evals/extract_probe.py.

Temperature is 0, but Ollama isn't bit-for-bit deterministic, so cases still run `runs` times.
"""

from collections import defaultdict

from evaluators import check_extraction, rate
from suites import load_jsonl


def run(runs: int) -> dict:
    from diary_agent.services.extraction import ExtractionError, build_extractor, extract_entry, get_extraction_model

    extractor = build_extractor(get_extraction_model())
    cases, failures = [], []
    by_check: dict[str, list[bool]] = defaultdict(list)
    for case in load_jsonl("extraction.jsonl"):
        for i in range(runs):
            try:
                entry = extract_entry(extractor, case["transcript"])
            except ExtractionError as e:
                cases.append({"id": case["id"], "run": i, "valid": False, "error": str(e)})
                by_check["valid"].append(False)
                failures.append(f"{case['id']:<26} run {i}: invalid ({e})")
                continue
            checks = check_extraction(case, entry)
            by_check["valid"].append(True)
            for name, ok in checks.items():
                by_check[name].append(ok)
            record = {"id": case["id"], "run": i, "valid": True, "checks": checks,
                      "entry": entry.model_dump(mode="json")}
            cases.append(record)
            if not all(checks.values()):
                bad = ", ".join(k for k, v in checks.items() if not v)
                failures.append(f"{case['id']:<26} run {i}: {bad} -> mood={entry.mood} words={entry.mood_words} "
                                f"hl={entry.highlights} ch={entry.challenges} gr={entry.gratitude} tags={entry.tags}")

    valid = [c for c in cases if c["valid"]]
    metrics = {"all_checks_pass": rate([all(c["checks"].values()) for c in valid] + [False] * (len(cases) - len(valid)))}
    metrics.update({name: rate(values) for name, values in by_check.items()})
    metrics["n_cases"] = len({c["id"] for c in cases})
    return {"metrics": metrics, "cases": cases, "failures": failures}
