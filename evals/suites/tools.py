"""Tool-use suite: one user turn per case through the *real* graph (router -> chat_agent <-> tools),
against tools_journal.jsonl in a temp dir, with "today" fixed so date answers are known.
Replaces M4's evals/tool_probe.py (same 14 cases, same checks, now in evaluators.check_tool_turn).

The chat model runs at its normal temperature (0.7), as in `diary chat`, so runs differ: use --runs 3.
Missed calls and false calls are reported separately, because one accuracy number hides which way
a model errs (M4: llama over-called, qwen3 under-called).

No agentevals: its trajectory match compares the list of tool calls, but the checks that caught
real bugs here are about the *arguments* (does the date range cover last Monday?), which a
custom check does better. A dependency should earn its place.
"""

import tempfile
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from evaluators import check_tool_turn, rate
from suites import load_jsonl, seed_journal

TODAY = date(2026, 10, 22)  # a Thursday; tools_journal.jsonl is written around it


def run(runs: int) -> dict:
    from langgraph.checkpoint.memory import InMemorySaver

    from diary_agent.core.config import settings
    from diary_agent.core.llm import get_chat_model
    from diary_agent.services.agent.graph import build_graph
    from diary_agent.services.agent.state import DiaryContext

    cases, failures, texts = [], [], []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Chroma may hold file locks
        conn, store = seed_journal("tools_journal.jsonl", Path(tmp))
        # No extractor or memory: one chat turn never reaches extraction, and no store = no memories.
        ctx = DiaryContext(chat_model=get_chat_model(), extractor=None, conn=conn, vectors=store,
                           day=TODAY, model_name=settings.chat_model)
        graph = build_graph(InMemorySaver())
        for case in load_jsonl("tools.jsonl"):
            for i in range(runs):
                config = {"configurable": {"thread_id": f"{case['id']}-{i}"}}  # fresh conversation each run
                started = time.perf_counter()
                state = graph.invoke({"user_input": case["text"]}, config, context=ctx)
                seconds = time.perf_counter() - started
                calls = [{"name": tc["name"], "args": tc["args"]}
                         for m in state["messages"] if m.type == "ai" for tc in m.tool_calls]
                reply = state["messages"][-1].content
                checks = check_tool_turn(case, calls, reply)
                cases.append({"id": case["id"], "run": i, "should_call": bool(case["tools"]), "calls": calls,
                              "reply": reply, "seconds": round(seconds, 2), "checks": checks})
                texts.append(reply)
                if not all(checks.values()):
                    bad = ", ".join(k for k, v in checks.items() if not v)
                    called = ", ".join(f"{c['name']}({c['args']})" for c in calls) or "no tools"
                    failures.append(f"{case['id']:<16} run {i}: {bad} -> {called} | {reply[:110]!r}")
        conn.close()

    by_check: dict[str, list[bool]] = defaultdict(list)
    for c in cases:
        for name, ok in c["checks"].items():
            by_check[name].append(ok)
    should, shouldnt = [c for c in cases if c["should_call"]], [c for c in cases if not c["should_call"]]
    metrics = {"all_checks_pass": rate([all(c["checks"].values()) for c in cases])}
    metrics.update({name: rate(values) for name, values in by_check.items()})
    metrics["missed_tool_calls"] = rate([not c["checks"]["tool"] for c in should])
    metrics["false_tool_calls"] = rate([not c["checks"]["tool"] for c in shouldnt])
    metrics["avg_tool_calls"] = sum(len(c["calls"]) for c in should) / len(should) if should else 0.0
    metrics["avg_seconds"] = sum(c["seconds"] for c in cases) / len(cases) if cases else 0.0
    metrics["n_cases"] = len({c["id"] for c in cases})
    return {"metrics": metrics, "cases": cases, "failures": failures, "texts": texts,
            "config": {"today": TODAY.isoformat()}}
