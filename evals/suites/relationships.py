"""Relationships suite: transcripts with gold people -> relationships.extract(), the real save-time
path (structured output, retries, guardrails). The save path has no human yes/no any more, so this
suite is what catches a wrong person before it ships.

Missed and invented people are reported separately, like missed and false tool calls: they are
different failures with different fixes (prompt recall vs guardrails).
"""

from collections import defaultdict

from evaluators import check_relationships, rate
from suites import load_jsonl


def run(runs: int) -> dict:
    from diary_agent.repositories.people_repository import normalise
    from diary_agent.services.extraction import ExtractionError, get_extraction_model
    from diary_agent.services.extractors import relationships

    runnable = relationships.build(get_extraction_model())
    cases, failures, texts = [], [], []
    by_check: dict[str, list[bool]] = defaultdict(list)
    missed = invented = 0
    for case in load_jsonl("relationships.jsonl"):
        expected = {normalise(p["name"]) for p in case["people"]}
        allowed = expected | {normalise(n) for n in case.get("allowed", [])}
        for i in range(runs):
            try:
                mentions = relationships.extract(runnable, case["transcript"])
            except ExtractionError as e:
                cases.append({"id": case["id"], "run": i, "valid": False, "error": str(e)})
                by_check["valid"].append(False)
                failures.append(f"{case['id']:<26} run {i}: invalid ({e})")
                continue
            checks = check_relationships(case, mentions)
            got = {normalise(m.name) for m in mentions}
            missed += len(expected - got)
            invented += len(got - allowed)
            texts.extend([m.what_happened for m in mentions] + [f for m in mentions for f in m.facts])
            by_check["valid"].append(True)
            for name, ok in checks.items():
                by_check[name].append(ok)
            cases.append({"id": case["id"], "run": i, "valid": True, "checks": checks,
                          "mentions": [m.model_dump() for m in mentions]})
            if not all(checks.values()):
                bad = ", ".join(k for k, v in checks.items() if not v)
                people = "; ".join(f"{m.name} ({m.relation}) {m.facts}" for m in mentions) or "nobody"
                failures.append(f"{case['id']:<26} run {i}: {bad} -> {people}")

    valid = [c for c in cases if c["valid"]]
    metrics = {"all_checks_pass": rate([all(c["checks"].values()) for c in valid] + [False] * (len(cases) - len(valid)))}
    metrics.update({name: rate(values) for name, values in by_check.items()})
    metrics["n_missed"] = missed
    metrics["n_invented"] = invented
    metrics["n_cases"] = len({c["id"] for c in cases})
    return {"metrics": metrics, "cases": cases, "failures": failures, "texts": texts}
