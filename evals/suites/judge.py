"""Judge suite: replies to chat.jsonl through the real graph, graded by the LLM judge (judge.py)
on the rubrics each case lists, plus the code check one_question on follow-up cases.

Not in the default run: about 40 judge calls per run on top of the replies. Use --only judge.
Two phases, replies first and then all the grading, so Ollama isn't asked to swap between the
agent's and the judge's settings on every case.

Only scores for checks in judge.CALIBRATED count. The others are printed as unverified: until a
judge's grades have been compared with yours (judge_calibrate.py), its numbers are a guess.
"""

import tempfile
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from evaluators import count_questions, rate
from judge import CALIBRATED, JUDGE_MODEL, RUBRIC_VERSION, build_judge, grade, view
from suites import load_jsonl, real_graph

TODAY = date(2026, 10, 22)  # same journal and day as the tools suite


def play(graph, ctx, case: dict, thread_id: str) -> dict:
    """Send each of the case's turns; return what the judge needs about the last reply."""
    config = {"configurable": {"thread_id": thread_id}}
    for text in case["turns"]:
        state = graph.invoke({"user_input": text}, config, context=ctx)
    messages = state["messages"]
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    lines = [f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in messages[:last_human + 1]
             if m.type == "human" or (m.type == "ai" and m.content and not m.tool_calls)]
    looked_up = "\n".join(m.content for m in messages[last_human:] if m.type == "tool")
    return {"conversation": "\n".join(lines), "looked_up": looked_up, "reply": messages[-1].content}


def generate(runs: int) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Chroma may hold file locks
        graph, ctx, conn = real_graph(Path(tmp), TODAY)
        for case in load_jsonl("chat.jsonl"):
            for i in range(runs):
                rows.append({"id": case["id"], "run": i, "checks": case["checks"],
                             **play(graph, ctx, case, f"{case['id']}-{i}")})
        conn.close()
    return rows


def run(runs: int) -> dict:
    rows = generate(runs)
    judge = build_judge()
    failures, seconds = [], []
    for row in rows:
        text = view(row["conversation"], row["looked_up"], row["reply"])
        row["verdicts"] = {}
        for check in row["checks"]:
            started = time.perf_counter()
            verdict = grade(judge, check, text)
            seconds.append(time.perf_counter() - started)
            row["verdicts"][check] = verdict.model_dump()
            if not verdict.passed:
                failures.append(f"{row['id']:<15} run {row['run']}: {check} - {verdict.reasoning[:120]!r}")
        if "follow_up" in row["checks"]:
            row["one_question"] = count_questions(row["reply"]) == 1
            if not row["one_question"]:
                failures.append(f"{row['id']:<15} run {row['run']}: one_question - "
                                f"{count_questions(row['reply'])} questions | {row['reply'][:90]!r}")

    by_check: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        for check, verdict in row["verdicts"].items():
            by_check[check].append(verdict["passed"])
    metrics = {check: rate(by_check[check]) for check in by_check}
    metrics["one_question"] = rate([r["one_question"] for r in rows if "one_question" in r])
    metrics["avg_judge_seconds"] = sum(seconds) / len(seconds) if seconds else 0.0
    metrics["n_replies"] = len(rows)
    unverified = [c for c in by_check if c not in CALIBRATED]
    if unverified:
        failures.insert(0, f"UNVERIFIED (judge not calibrated, treat as a guess): {', '.join(unverified)}")
    return {"metrics": metrics, "cases": rows, "failures": failures, "texts": [r["reply"] for r in rows],
            "config": {"today": TODAY.isoformat(), "judge_model": JUDGE_MODEL, "rubric_version": RUBRIC_VERSION,
                       "calibrated": CALIBRATED}}
