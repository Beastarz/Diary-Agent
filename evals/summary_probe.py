"""M5 probe: are week/month summaries grounded, and what does each map-reduce stage cost?

Run: uv run python evals/summary_probe.py [--runs 2] [--collapse]

Uses tool_probe's seeded 10-entry October journal (summaries + fields, no transcripts).
--collapse shrinks the reduce budget so the collapse step has to run, to see what merging loses.
Checks are deterministic and crude (an LLM judge comes in M7):
- grounded:  every win/challenge shares a content word with some entry it summarizes
- recurring: themes mention a tag that really recurred (work, family, health)
- mood:      mood_trend names both the lowest and the highest day (weekday, date or event)
- embellish: the review doesn't claim something finished that the entry says only started (pottery)
- sparse:    with 2 entries in a week, no "consistently" / "throughout the week" claims
- clean:     no CJK characters
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

from tool_probe import CJK, D, seed

STOP = {"the", "a", "an", "and", "with", "your", "you", "of", "to", "at", "on", "in", "for", "about", "new", "got"}
CASES = [
    {"id": "week_oct12", "period": "week", "day": D(14), "low": ["thursday", "cold", "sick"],
     "high": ["saturday", "10k", "personal best"], "recurring": []},
    {"id": "month_oct", "period": "month", "day": D(22), "low": ["thursday", "cold", "sick", "15"],
     "high": ["saturday", "10k", "personal best", "17"], "recurring": ["work", "family", "health"]},
    # 2 entries in 7 days (pottery Thu, hike Sat, both mood 7): no low/high to name, nothing recurs.
    {"id": "sparse_week_oct5", "period": "week", "day": D(8), "sparse": True, "recurring": []},
]
# Claims about the whole period. Fine for a full month; wrong for 2 days out of 7.
OVERCLAIM = re.compile(r"\b(consistent|steady|steadily|throughout|all week|every day|whole week)", re.IGNORECASE)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


FINISHED = re.compile(r"\b(complet|finish)", re.IGNORECASE)


def check(case: dict, record, entries) -> dict[str, bool]:
    source = set().union(*(words(" ".join([e.summary, *e.highlights, *e.challenges])) for e in entries))
    items = record.wins + record.challenges
    trend = record.mood_trend.lower()
    text = " ".join([record.overview, record.mood_trend, *record.themes, *items])
    res = {
        "grounded": all(words(i) & source for i in items),
        "clean": not CJK.search(text),
        "embellish": not any(FINISHED.search(i) for i in [*items, record.overview] if "pottery" in i.lower()),
    }
    if case.get("sparse"):
        res["sparse"] = not OVERCLAIM.search(f"{record.overview} {record.mood_trend}")
    else:
        res["mood"] = any(k in trend for k in case["low"]) and any(k in trend for k in case["high"])
    if case["recurring"]:
        res["recurring"] = any(t in " ".join(record.themes).lower() for t in case["recurring"])
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--collapse", action="store_true", help="force the collapse step with a tiny budget")
    args = parser.parse_args()

    from langchain_core.embeddings import DeterministicFakeEmbedding

    from diary_agent.core.config import settings
    from diary_agent.core.prompts import PROMPT_VERSION
    from diary_agent.models.summary import PeriodSummary
    from diary_agent.repositories import entry_repository as entry_repo
    from diary_agent.services import summary as svc

    if args.collapse:
        svc.REDUCE_INPUT_TOKENS = 150
        svc.collapse.__defaults__ = (150,)

    from langchain_core.runnables import RunnableLambda

    from diary_agent.core.prompts import COLLAPSE_SYSTEM

    # Spy on the real summarize_period: count calls per stage and keep the reduce prompt
    # (the code-built day lines + stats) to see its size and what the model was given.
    calls: list[tuple[str, str]] = []  # (stage, prompt text)

    def spy(runnable, reduce_step: bool):
        def fn(messages):
            out = runnable.invoke(messages)
            stage = "reduce" if reduce_step else "collapse" if messages[0].content == COLLAPSE_SYSTEM else "other"
            calls.append((stage, messages[1].content))
            return out

        return RunnableLambda(fn)

    writer = spy(svc.build_writer(svc.get_summary_model(max_tokens=200)), reduce_step=False)
    reducer = spy(svc.build_reducer(svc.get_summary_model(max_tokens=800)), reduce_step=True)

    records = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Chroma may hold file locks
        conn, _ = seed(Path(tmp), DeterministicFakeEmbedding(size=8))  # vectors unused here
        for case in CASES:
            start, end = svc.period_range(case["period"], case["day"])
            entries = entry_repo.list_entries(conn, start=start, end=end)
            for i in range(args.runs):
                calls.clear()
                t0 = time.perf_counter()
                record = svc.summarize_period(writer, reducer, entries, case["period"], start, end, settings.chat_model)
                seconds = time.perf_counter() - t0
                counts = {s: sum(1 for c, _ in calls if c == s) for s in ("collapse", "reduce")}
                reduce_prompt = next(text for stage, text in reversed(calls) if stage == "reduce")
                checks = check(case, record, entries)
                summary = record.model_dump(include=set(PeriodSummary.model_fields))
                records.append({
                    "case": case["id"], "run": i, "checks": checks, "calls": counts,
                    "reduce_prompt_tokens": svc.estimate_tokens(reduce_prompt),
                    "seconds": round(seconds, 1), "summary": summary, "reduce_prompt": reduce_prompt,
                })
                print(f"\n--- {case['id']} run {i}: {len(entries)} entries, calls {counts}, "
                      f"reduce prompt ~{svc.estimate_tokens(reduce_prompt)} tokens, {seconds:.1f}s ---")
                print(json.dumps(summary, indent=2))
                print("checks:", checks)

    print("\n=== totals ===")
    keys = sorted({k for r in records for k in r["checks"]})
    for k in keys:
        vals = [r["checks"][k] for r in records if k in r["checks"]]
        print(f"{k:<10} {sum(vals)}/{len(vals)}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"summary_probe-{stamp}.json"
    out.write_text(json.dumps({"model": settings.chat_model, "prompt_version": PROMPT_VERSION, "commit": commit,
                               "collapse_forced": args.collapse, "today": str(date.today()), "records": records},
                              indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
