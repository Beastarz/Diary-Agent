"""Language suite: no model calls of its own. It scans every text the other suites produced in
this run (tool-turn replies, extracted entries) for CJK characters: the M1 leak, where qwen3 mixed
Chinese tokens into English replies. Formalizes M1's first eval at zero cost.

Only as wide as what the other suites generated: run it with tools and extraction. Replies to
non-English messages (does it answer in the user's language?) would need cases of their own.
"""

from evaluators import has_cjk, rate


def run_on(texts_by_suite: dict[str, list[str]]) -> dict:
    cases, failures = [], []
    for suite, texts in texts_by_suite.items():
        for i, text in enumerate(texts):
            leak = has_cjk(text)
            cases.append({"suite": suite, "index": i, "cjk": leak})
            if leak:
                failures.append(f"{suite} #{i}: {text[:110]!r}")
    metrics = {"no_cjk": rate([not c["cjk"] for c in cases]), "n_texts": len(cases)}
    return {"metrics": metrics, "cases": cases, "failures": failures, "sources": sorted(texts_by_suite)}
