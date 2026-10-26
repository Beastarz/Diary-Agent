"""Routing suite: no LLM, so it's instant and deterministic (runs is ignored).

Two code decisions are measured, both made before any model sees the message:
- route: what nodes.router does with the message (chat, checkin, extract, summarize, done)
- tools: which tools nodes.tools_for offers in chat (none, read-only, or read + remember_fact)

This is the baseline for the LLM router fallback deferred since M3: build it only if it fixes
the misses below without breaking the rest.
"""

from evaluators import confusion, rate
from suites import load_jsonl


def run(runs: int) -> dict:
    from diary_agent.services.agent.nodes import router, tools_for
    from diary_agent.services.agent.tools import READ_TOOLS, TOOLS

    menu = {0: "none", len(READ_TOOLS): "read", len(TOOLS): "all"}
    cases, failures = [], []
    for case in load_jsonl("routing.jsonl"):
        state = {"user_input": case["text"], "mode": case.get("mode", "chat"), "checkin_step": case.get("step", 0)}
        route = router(state)["route"]
        tools = menu[len(tools_for(case["text"]))] if route == "chat" else None
        record = {**case, "got_route": route, "got_tools": tools, "route_ok": route == case["route"]}
        if "tools" in case:
            record["tools_ok"] = record["route_ok"] and tools == case["tools"]
        cases.append(record)
        if not record["route_ok"] or not record.get("tools_ok", True):
            got = route + (f"/{tools}" if tools else "")
            want = case["route"] + (f"/{case['tools']}" if "tools" in case else "")
            known = f"  [known since {case['known']}]" if "known" in case else ""
            failures.append(f"{case['id']:<26} {case['text']!r}: got {got}, want {want}{known}")

    with_tools = [c for c in cases if "tools" in c]
    metrics = {
        "route_accuracy": rate([c["route_ok"] for c in cases]),
        "tool_gate_accuracy": rate([c["tools_ok"] for c in with_tools]),
        "tools_offered_when_not_needed": rate([c["got_tools"] not in (None, "none") for c in with_tools
                                               if c["tools"] == "none"]),
        "tools_missing_when_needed": rate([c["got_tools"] in (None, "none") for c in with_tools
                                           if c["tools"] != "none"]),
        "n_cases": len(cases),
    }
    return {"metrics": metrics, "cases": cases, "failures": failures,
            "confusion": confusion([(c["route"], c["got_route"]) for c in cases])}
