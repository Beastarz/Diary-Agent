"""Calibrate the LLM judge against your own grades, before trusting any judge score.

  uv run python evals/judge_calibrate.py sample          # writes datasets/judge_gold.csv to grade
  (grade it: y = pass, n = fail in every empty cell; n/a cells don't apply; note is optional)
  uv run python evals/judge_calibrate.py agree --runs 3  # judge grades the same rows; compare

The grading file holds one llama3.1 reply per chat.jsonl case plus the hand-written bad replies
from judge_planted.jsonl, shuffled, so you grade blind. Replies are frozen in the file: once graded,
it's a fixed test of the *judge*, and can be re-used for any judge model or rubric version.

agree prints, per check: plain agreement, fail_catch (of your fails, how many the judge caught),
false_alarm (of your passes, how many it failed) and consistent (same verdict on every repeat).
Rule of thumb for putting a check on the scorecard (judge.CALIBRATED): agree >= 85% and
fail_catch >= 80%. Below that, the judge's number for that check is noise with a decimal point.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from evaluators import agreement
from judge import CHECKS, JUDGE_MODEL, RUBRIC_VERSION, build_judge, grade, view
from suites import DATASETS, load_jsonl

GOLD = DATASETS / "judge_gold.csv"
RESULTS = Path(__file__).parent / "results"
COLUMNS = ("row", "case", "conversation", "looked_up", "reply", *CHECKS, "note")
GRADES = {"y": True, "yes": True, "pass": True, "n": False, "no": False, "fail": False}


def read_gold() -> list[dict]:
    with GOLD.open(encoding="utf-8-sig", newline="") as f:  # -sig: Excel writes a BOM
        return list(csv.DictReader(f))


def sample(force: bool) -> None:
    from suites.judge import generate

    if GOLD.exists() and not force and any(r[c].strip().lower() in GRADES for r in read_gold() for c in CHECKS):
        sys.exit(f"{GOLD.name} already has grades in it. Use --force to overwrite them.")
    generated = {r["id"]: r for r in generate(runs=1)}
    rows = list(generated.values())
    for p in load_jsonl("judge_planted.jsonl"):
        rows.append({**generated[p["case"]], "reply": p["reply"]})  # same conversation, worse reply
    random.Random(7).shuffle(rows)  # planted replies don't sit next to their originals

    with GOLD.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        for n, r in enumerate(rows, start=1):
            writer.writerow({"row": n, "case": r["id"], "conversation": r["conversation"],
                             "looked_up": r["looked_up"] or "(nothing looked up)", "reply": r["reply"],
                             **{c: "" if c in r["checks"] else "n/a" for c in CHECKS}, "note": ""})
    cells = sum(c in r["checks"] for r in rows for c in CHECKS)
    print(f"wrote {GOLD} ({len(rows)} replies, {cells} cells to grade with y/n)")


def agree(runs: int) -> None:
    rows = read_gold()
    missing = [f"row {r['row']} {c}" for r in rows for c in CHECKS
               if r[c].strip().lower() != "n/a" and r[c].strip().lower() not in GRADES]
    if missing:
        sys.exit("Not graded yet (use y or n): " + ", ".join(missing))

    judge = build_judge()
    cells, pairs = [], {c: [] for c in CHECKS}
    for r in rows:
        text = view(r["conversation"], "" if r["looked_up"] == "(nothing looked up)" else r["looked_up"], r["reply"])
        for check in CHECKS:
            if r[check].strip().lower() == "n/a":
                continue
            human = GRADES[r[check].strip().lower()]
            verdicts = [grade(judge, check, text) for _ in range(runs)]
            votes = [v.passed for v in verdicts]
            judged = sum(votes) * 2 > len(votes)  # majority; a tie counts as fail
            pairs[check].append((human, judged))
            cells.append({"row": r["row"], "case": r["case"], "check": check, "human": human, "judge": judged,
                          "consistent": len(set(votes)) == 1, "reasoning": [v.reasoning for v in verdicts]})
            if human != judged:
                print(f"  row {r['row']:>2} {r['case']:<15} {check:<10} you={'pass' if human else 'fail'} "
                      f"judge={'pass' if judged else 'fail'}: {verdicts[0].reasoning[:110]!r}")

    print(f"\njudge={JUDGE_MODEL} rubric={RUBRIC_VERSION} runs={runs}")
    print(f"  {'check':<10} {'n':>3} {'fails':>5} {'agree':>6} {'catch':>6} {'alarm':>6} {'consist':>8}")
    summary = {}
    for check in CHECKS:
        if not pairs[check]:
            continue
        a = agreement(pairs[check])
        a["consistent"] = sum(c["consistent"] for c in cells if c["check"] == check) / a["n_graded"]
        summary[check] = a
        ok = a["agree"] >= 0.85 and a["fail_catch"] >= 0.8 and a["n_human_fails"] >= 2
        print(f"  {check:<10} {a['n_graded']:>3} {a['n_human_fails']:>5} {a['agree']:>6.0%} {a['fail_catch']:>6.0%} "
              f"{a['false_alarm']:>6.0%} {a['consistent']:>8.0%}   {'trust' if ok else 'not yet'}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"judge-calibration-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"meta": {"judge_model": JUDGE_MODEL, "rubric_version": RUBRIC_VERSION, "runs": runs,
                                         "commit": commit, "started": datetime.now().isoformat()},
                                "summary": summary, "cells": cells}, indent=2), encoding="utf-8")
    print(f"\nsaved {path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="write the hand-grading file")
    s.add_argument("--force", action="store_true", help="overwrite a file that already has grades")
    a = sub.add_parser("agree", help="judge the graded rows and compare")
    a.add_argument("--runs", type=int, default=3, help="judge repeats per cell (majority vote)")
    args = parser.parse_args()
    sample(args.force) if args.cmd == "sample" else agree(args.runs)


if __name__ == "__main__":
    main()
