"""M7 eval harness: one command, one scorecard, one results file per run.

Run: uv run python evals/run_evals.py [--only routing,retrieval,extraction] [--runs 2]

Suites (cheapest first):
- routing     no LLM. Labelled messages -> the router's route, and which tools the code gate offers.
- retrieval   embeddings only. Seeded 60-entry journal + questions with known answer days.
- extraction  chat model. Transcripts with gold labels -> extract_entry().

Results go to evals/results/evals-<timestamp>.json with the model, prompt version and git commit.
The scorecard shows each metric's change vs the latest earlier run of the same suite, so the loop
is: run, change one thing, run again, keep the change only if the numbers say so.
"""

import argparse
import importlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

SUITES = ("routing", "retrieval", "extraction")
RESULTS = Path(__file__).parent / "results"


def previous_metrics(suite: str) -> tuple[dict, str] | tuple[None, None]:
    for path in sorted(RESULTS.glob("evals-*.json"), reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        if suite in data["suites"]:
            return data["suites"][suite]["metrics"], path.stem.removeprefix("evals-")
    return None, None


def kind(name: str) -> str:
    """Naming rule for metrics: n_* are counts, *mrr / *distance are plain numbers, the rest are rates."""
    if name.startswith("n_"):
        return "count"
    return "number" if name.endswith(("mrr", "distance")) else "rate"


def fmt(name: str, value: float, signed: bool = False) -> str:
    sign = "+" if signed and value > 0 else ""
    return {"count": f"{sign}{value:.0f}", "number": f"{sign}{value:.2f}"}.get(kind(name), f"{sign}{value * 100:.0f}%")


def scorecard(suite: str, metrics: dict) -> None:
    before, when = previous_metrics(suite)
    print(f"\n=== {suite} ===" + (f"   (vs {when})" if when else "   (first run)"))
    for name, value in metrics.items():
        changed = before and name in before and abs(before[name] - value) > 1e-9
        delta = f"   {fmt(name, value - before[name], signed=True)}" if changed else ""
        print(f"  {name:<30} {fmt(name, value):>6}{delta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(SUITES), help="comma-separated suites")
    parser.add_argument("--runs", type=int, default=2, help="repeats for LLM suites (outputs vary)")
    args = parser.parse_args()
    chosen = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(chosen) - set(SUITES)
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(sorted(unknown))}")

    from diary_agent.core.config import settings
    from diary_agent.core.prompts import PROMPT_VERSION

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    meta = {"model": settings.chat_model, "embed_model": settings.embed_model, "prompt_version": PROMPT_VERSION,
            "commit": commit + ("+dirty" if dirty else ""), "runs": args.runs, "started": datetime.now().isoformat()}
    print(f"model={meta['model']} embed={meta['embed_model']} prompt={PROMPT_VERSION} commit={meta['commit']}")

    results = {"meta": meta, "suites": {}}
    for suite in chosen:
        out = importlib.import_module(f"suites.{suite}").run(args.runs)
        scorecard(suite, out["metrics"])  # before saving, so the delta is vs the *previous* run
        for line in out["failures"]:
            print(f"    x {line}")
        results["suites"][suite] = out

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"evals-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
