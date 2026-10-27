"""Simulation suite: a simulated user (simulator.py) runs a full /checkin with the real graph, the
code approves the save, and the entry that lands in journal.db is checked against the persona's
script. The only suite that tests the whole loop: check-in -> extract -> confirm -> save.

Per conversation:
- completed:     an entry was saved within MAX_ANSWERS answers (the check-in needs 4)
- in_character:  every simulated turn passed evaluators.in_character; if not, the run is dropped
                 from the entry and judge scores (it would measure the simulator, not the agent)
- entry checks:  check_extraction against the persona's gold (mood, mentions, placement, ...), limited
                 to facts the simulator actually said (evaluators.said_gold)
- judge:         follow_up and grounded on each check-in question after the first, plus the code
                 check one_question (unverified until judge.CALIBRATED says otherwise)

Not in the default run: ~4 agent turns, ~3 simulator turns, an extraction and ~6 judge calls per
conversation. Use --only simulation.
"""

import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

from evaluators import check_extraction, count_questions, in_character, rate, said_gold
from judge import CALIBRATED, JUDGE_MODEL, RUBRIC_VERSION, build_judge, grade, view
from simulator import SIM_MODEL, build_sim, sim_reply
from suites import load_jsonl, real_graph

TODAY = date(2026, 10, 22)  # same journal and day as the tools and judge suites
MAX_ANSWERS = 6  # the check-in asks 4 questions; more means it got stuck
JUDGED = ("follow_up", "grounded")


def render(transcript: list[tuple[str, str]]) -> str:
    return "\n".join(f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in transcript)


def converse(graph, ctx, sim, persona: dict, thread_id: str) -> dict:
    """One whole check-in. Returns the transcript (as the user saw it), the saved entry and timing."""
    from langgraph.types import Command

    from diary_agent.repositories import entry_repository as entry_repo
    from diary_agent.services.agent.graph import pending_interrupt

    with ctx.conn:  # each run starts with no entry for today, so a failed save can't hide behind an old one
        ctx.conn.execute("DELETE FROM entries WHERE entry_date = ?", (ctx.day.isoformat(),))
    config = {"configurable": {"thread_id": thread_id}}
    started = time.perf_counter()
    state = graph.invoke({"user_input": "/checkin"}, config, context=ctx)
    transcript = [("user", "Let's do a check-in.")]  # what the router turns /checkin into
    injected = {int(k): v for k, v in persona.get("inject", {}).items()}
    sim_turns = []
    for i in range(MAX_ANSWERS):
        # Paused on the save question, or no new reply (extraction failed and left a notice): done.
        if pending_interrupt(graph, config) is not None or state["messages"][-1].type != "ai":
            break
        transcript.append(("assistant", state["messages"][-1].content))
        answer = injected.get(i)
        if answer is None:
            answer = sim_reply(sim, persona, transcript[1:])
            sim_turns.append(answer)
        transcript.append(("user", answer))
        state = graph.invoke({"user_input": answer}, config, context=ctx)
    payload = pending_interrupt(graph, config)
    if payload is not None and "draft" in payload:
        graph.invoke(Command(resume=True), config, context=ctx)  # the human says yes
    entry = entry_repo.get_entry(ctx.conn, ctx.day)
    return {"transcript": transcript, "entry": entry, "seconds": time.perf_counter() - started, "sim_turns": sim_turns}


def run(runs: int) -> dict:
    from diary_agent.services.extraction import build_extractor, get_extraction_model

    sim, judge = build_sim(), build_judge()
    convos, failures, texts = [], [], []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Chroma may hold file locks
        graph, ctx, conn = real_graph(Path(tmp), TODAY)
        ctx = replace(ctx, extractor=build_extractor(get_extraction_model()))
        for persona in load_jsonl("personas.jsonl"):
            for i in range(runs):
                out = converse(graph, ctx, sim, persona, f"{persona['id']}-{i}")
                entry, transcript = out["entry"], out["transcript"]
                record = {"id": persona["id"], "run": i, "transcript": transcript, "seconds": round(out["seconds"], 1),
                          "completed": entry is not None, "in_character": all(map(in_character, out["sim_turns"]))}
                tag = f"{persona['id']:<15} run {i}:"
                if not record["in_character"]:
                    bad = next(t for t in out["sim_turns"] if not in_character(t))
                    failures.append(f"{tag} DROPPED, simulator out of character: {bad[:100]!r}")
                    convos.append(record)
                    continue
                if entry is None:
                    failures.append(f"{tag} no entry saved after {len(transcript) // 2} answers")
                else:
                    record["entry"] = entry.model_dump(mode="json", include={"mood", "highlights", "challenges",
                                                                              "gratitude", "mood_words", "summary"})
                    said = " ".join(t for role, t in transcript if role == "user")
                    gold, record["gold_unsaid"] = said_gold(persona["gold"], said)
                    record["checks"] = check_extraction(gold, entry)
                    texts.append(" ; ".join([entry.summary, *entry.highlights, *entry.challenges, *entry.gratitude]))
                    bad = [k for k, v in record["checks"].items() if not v]
                    if bad:
                        failures.append(f"{tag} entry {', '.join(bad)} -> {record['entry']}")
                # Grade each check-in question that answers something the user said (not the first,
                # which replies to "Let's do a check-in.").
                record["verdicts"] = []
                for j, (role, reply) in enumerate(transcript):
                    if role != "assistant" or j < 3:
                        continue
                    texts.append(reply)
                    text = view(render(transcript[:j]), "", reply)
                    verdicts = {c: grade(judge, c, text).model_dump() for c in JUDGED}
                    verdicts["one_question"] = count_questions(reply) == 1
                    record["verdicts"].append(verdicts)
                    for c in JUDGED:
                        if not verdicts[c]["passed"]:
                            failures.append(f"{tag} turn {j // 2} {c} - {verdicts[c]['reasoning'][:110]!r}")
                convos.append(record)
        conn.close()

    usable = [c for c in convos if c["in_character"]]
    by_check: dict[str, list[bool]] = defaultdict(list)
    for c in usable:
        for name, ok in c.get("checks", {}).items():
            by_check[name].append(ok)
        for v in c["verdicts"]:
            for name in JUDGED:
                by_check[f"turn_{name}"].append(v[name]["passed"])
            by_check["turn_one_question"].append(v["one_question"])
    saved = [c for c in usable if c["completed"]]
    metrics = {"completed": rate([c["completed"] for c in usable]),
               "entry_all_pass": rate([all(c["checks"].values()) for c in saved]),
               **{name: rate(values) for name, values in by_check.items()},
               "sim_in_character": rate([c["in_character"] for c in convos]),
               # Gold facts the simulator never said, so they weren't scored. High = the simulator forgets.
               "avg_gold_unsaid": sum(c.get("gold_unsaid", 0) for c in saved) / len(saved) if saved else 0.0,
               "avg_seconds": sum(c["seconds"] for c in convos) / len(convos) if convos else 0.0,
               "n_conversations": len(convos)}
    unverified = [c for c in JUDGED if c not in CALIBRATED]
    if unverified:
        failures.insert(0, f"UNVERIFIED (judge not calibrated, treat as a guess): {', '.join('turn_' + c for c in unverified)}")
    return {"metrics": metrics, "cases": convos, "failures": failures, "texts": texts,
            "config": {"today": TODAY.isoformat(), "sim_model": SIM_MODEL, "judge_model": JUDGE_MODEL,
                       "rubric_version": RUBRIC_VERSION, "max_answers": MAX_ANSWERS}}
