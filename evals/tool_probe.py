"""M4 eval: does the model call the right tool, with the right arguments, and stay grounded?

Run: uv run python evals/tool_probe.py [--runs 3] [--models llama3.1:8b other:tag]

Each run is one user turn through the *real* graph (router -> chat_agent <-> tools), against a
seeded 10-entry journal in a temp dir, with "today" fixed to TODAY so date answers are known.
Checks are deterministic:
- tool:     an acceptable tool was called (or, for "none" cases, no tool was called)
- args:     some call had valid arguments; for date questions its range covers the right days
            and none of the wrong ones
- answer:   the final reply mentions a fact from the right entry (any keyword matches).
            For the "missing" case: the reply admits nothing was found. Keyword checks are crude;
            an LLM judge replaces them in M7.
- clean:    no CJK characters in the reply (the M1 leak)
Plus a model-free retrieval check: is the gold entry in the top-k vector search results?
That separates "search is bad" from "the model used it badly".
Results go to evals/results/ as JSON, tagged with model and prompt version.
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

TODAY = date(2026, 10, 22)  # a Thursday
D = lambda day: date(2026, 10, day)  # noqa: E731

# (day, summary, highlights, challenges, tags, mood)
SEED = [
    (8, "You started a pottery class and made a wonky bowl.", ["first pottery class", "made a bowl"], [], ["hobby"], 7),
    (10, "You hiked Bukit Timah with Mei and got caught in the rain at the top.", ["hike with Mei"], ["rain"], ["outdoors", "friends"], 7),
    (12, "Your big work presentation went well and your manager praised you.", ["presentation went well", "manager praised you"], ["nerves before presenting"], ["work"], 8),
    (13, "You argued with your roommate about the dishes and felt stressed.", [], ["argument with roommate about dishes"], ["home", "stress"], 4),
    (14, "You had dinner with your sister at a new ramen place.", ["ramen dinner with sister"], [], ["family", "food"], 8),
    (15, "You were sick with a cold and stayed home all day.", [], ["cold", "stayed home"], ["health"], 3),
    (17, "You ran a 10k and set a personal best.", ["10k personal best"], [], ["running", "health"], 9),
    (19, "You worked late on the Q4 report because of a deadline.", [], ["Q4 report deadline", "worked late"], ["work"], 4),
    (20, "You called your grandma and she told stories about her childhood.", ["call with grandma"], [], ["family"], 8),
    (21, "You went to the dentist, then bought a monstera plant.", ["new monstera plant"], ["dentist appointment"], ["errands"], 6),
]

DATE_TOOLS = {"get_entries_by_date"}
CASES = [
    # --- date questions: the model has to do the calendar math ---
    {"id": "date_yesterday", "text": "What did I do yesterday?",
     "tools": DATE_TOOLS, "cover": [D(21)], "exclude": [D(20), D(22)], "answer": ["dentist", "monstera", "plant"]},
    {"id": "date_specific", "text": "What did I write on October 12?",
     "tools": DATE_TOOLS, "cover": [D(12)], "exclude": [D(13)], "answer": ["presentation"]},
    {"id": "date_monday", "text": "How was my Monday this week?",
     "tools": DATE_TOOLS, "cover": [D(19)], "exclude": [D(12), D(21)], "answer": ["q4", "report", "deadline", "late"]},
    {"id": "date_tuesday", "text": "What happened last Tuesday?",  # ambiguous: Oct 20 or Oct 13, both accepted
     "tools": DATE_TOOLS, "cover_any": [D(20), D(13)], "exclude": [D(21), D(22)],
     "answer": ["grandma", "roommate", "dishes"]},
    {"id": "date_last_week", "text": "How was my week last week?",  # Mon Oct 12 - Sun Oct 18
     "tools": DATE_TOOLS, "cover": [D(12), D(17)], "exclude": [D(8), D(20)],
     "answer": ["presentation", "roommate", "ramen", "sister", "cold", "10k"]},
    # --- topic questions: no date given, semantic search is the right tool ---
    {"id": "topic_sister", "text": "When did I last see my sister?", "tools": {"search_entries"},
     "gold": D(14), "answer": ["ramen", "14"]},
    {"id": "topic_pottery", "text": "What did I make at my pottery class?", "tools": {"search_entries"},
     "gold": D(8), "answer": ["bowl"]},
    {"id": "topic_running", "text": "Have I set any running records recently?", "tools": {"search_entries"},
     "gold": D(17), "answer": ["10k", "10 k", "personal best"]},
    {"id": "topic_roommate", "text": "Did I have any fights with my roommate?", "tools": {"search_entries"},
     "gold": D(13), "answer": ["dishes"]},
    {"id": "topic_missing", "text": "When did I last go skiing?", "tools": {"search_entries", "get_entries_by_date"},
     "admits": True},  # never happened: the reply must not invent a ski trip
    # --- no tool: small talk, or the user telling us about today ---
    {"id": "none_greeting", "text": "hey! how's it going?", "tools": set()},
    {"id": "none_today", "text": "Today I had a really long meeting and I'm exhausted.", "tools": set()},
    {"id": "none_feeling", "text": "I'm feeling kind of anxious tonight.", "tools": set()},
    {"id": "none_thanks", "text": "thanks, that actually helps", "tools": set()},
]

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
ADMITS = ("couldn't", "could not", "can't find", "cannot find", "didn't find", "no entries", "no entry",
          "nothing", "don't see", "do not see", "no record", "not find", "no mention", "doesn't look like",
          "don't have", "no matching", "don't mention", "doesn't mention", "not mention", "haven't",
          "no sign", "isn't any", "there's no")


def _covers(call: dict, case: dict) -> bool:
    args = call["args"]
    try:
        start = date.fromisoformat(args["start_date"]) if args.get("start_date") else None
        end = date.fromisoformat(args["end_date"]) if args.get("end_date") else None
    except (ValueError, TypeError):
        return False
    if start is None and end is None:
        return False  # a date question answered by an unfiltered search
    if call["name"] == "get_entries_by_date":
        end = end or start
    lo, hi = start or date.min, end or date.max
    inside = lambda d: lo <= d <= hi  # noqa: E731
    if "cover" in case and not all(inside(d) for d in case["cover"]):
        return False
    if "cover_any" in case and not any(inside(d) for d in case["cover_any"]):
        return False
    return not any(inside(d) for d in case.get("exclude", []))


def check(case: dict, calls: list[dict], reply: str) -> dict[str, bool]:
    names = {c["name"] for c in calls}
    text = reply.lower()
    res = {"tool": names <= case["tools"] and bool(names) if case["tools"] else not names}
    if case["tools"]:
        relevant = [c for c in calls if c["name"] in case["tools"]]
        if "cover" in case or "cover_any" in case:
            res["args"] = any(_covers(c, case) for c in relevant)
        else:
            res["args"] = any(c["args"].get("query", "").strip() for c in relevant if c["name"] == "search_entries") \
                or any(c["name"] == "get_entries_by_date" for c in relevant)
    if "answer" in case:
        res["answer"] = any(k in text for k in case["answer"])
    if case.get("admits"):
        res["answer"] = any(k in text for k in ADMITS)
    res["clean"] = not CJK.search(reply)
    return res


def seed(tmp: Path, embeddings):
    from diary_agent.core.prompts import PROMPT_VERSION
    from diary_agent.models.entry import JournalEntry
    from diary_agent.repositories import entry_repository as entry_repo
    from diary_agent.repositories import vector_repository as vector_repo

    conn = entry_repo.connect(tmp / "journal.db")
    store = vector_repo.open_store(tmp / "chroma", embeddings)
    for day, summary, highlights, challenges, tags, mood in SEED:
        entry = JournalEntry(entry_date=D(day), mood=mood, mood_words=[], highlights=highlights,
                             challenges=challenges, gratitude=[], tags=tags, summary=summary,
                             transcript="", model="seed", prompt_version=PROMPT_VERSION)
        entry_repo.save_entry(conn, entry)
        vector_repo.upsert_entry(store, entry)
    return conn, store


def retrieval_check(store) -> dict:
    """Model-free: for each topic case, is the gold day in the top-k for the user's own words?"""
    from diary_agent.repositories import vector_repository as vector_repo
    from diary_agent.services.agent.tools import SEARCH_K

    out = {}
    for case in (c for c in CASES if "gold" in c):
        days = [d for d, _ in vector_repo.search(store, case["text"], k=SEARCH_K)]
        rank = days.index(case["gold"]) + 1 if case["gold"] in days else None
        out[case["id"]] = rank
    return out


def run_model(model_name: str, runs: int, conn, store) -> list[dict]:
    from langchain_ollama import ChatOllama
    from langgraph.checkpoint.memory import InMemorySaver

    from diary_agent.core.config import settings
    from diary_agent.services.agent.graph import build_graph
    from diary_agent.services.agent.state import DiaryContext

    model = ChatOllama(
        model=model_name, base_url=settings.ollama_base_url, temperature=settings.temperature,
        reasoning=settings.reasoning or None,  # same as core/llm.py
    )
    # extractor=None: one chat turn never reaches extraction.
    ctx = DiaryContext(chat_model=model, extractor=None, conn=conn, vectors=store, day=TODAY, model_name=model_name)
    graph = build_graph(InMemorySaver())
    records = []
    for case in CASES:
        for i in range(runs):
            config = {"configurable": {"thread_id": f"{case['id']}-{i}"}}  # fresh conversation each run
            started = time.perf_counter()
            state = graph.invoke({"user_input": case["text"]}, config, context=ctx)
            seconds = time.perf_counter() - started
            calls = [{"name": tc["name"], "args": tc["args"]}
                     for m in state["messages"] if m.type == "ai" for tc in m.tool_calls]
            reply = state["messages"][-1].content
            records.append({"model": model_name, "case": case["id"], "run": i, "calls": calls, "reply": reply,
                            "seconds": round(seconds, 2), "checks": check(case, calls, reply)})
    return records


def report(model_name: str, records: list[dict], runs: int) -> dict:
    print(f"\n=== {model_name} ===")
    by_case = defaultdict(list)
    for r in records:
        by_case[r["case"]].append(r)
    totals: dict[str, list[bool]] = defaultdict(list)
    for case in CASES:
        rs = by_case[case["id"]]
        keys = list(rs[0]["checks"])
        cells = "  ".join(f"{k}={sum(r['checks'][k] for r in rs)}/{runs}" for k in keys)
        print(f"{case['id']:<16} {cells}")
        for r in rs:
            for k, v in r["checks"].items():
                totals[k].append(v)
        if not all(all(r["checks"].values()) for r in rs):  # show one failing sample
            bad = next(r for r in rs if not all(r["checks"].values()))
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in bad["calls"]) or "no tools"
            print(f"{'':<16} -> {calls} | {bad['reply'][:140]!r}")
    should_call = [r for r in records if next(c for c in CASES if c["id"] == r["case"])["tools"]]
    no_call = [r for r in records if r not in should_call]
    summary = {k: round(sum(v) / len(v), 3) for k, v in totals.items()}
    summary["missed_tool_rate"] = round(sum(not r["checks"]["tool"] for r in should_call) / len(should_call), 3)
    summary["false_tool_rate"] = round(sum(not r["checks"]["tool"] for r in no_call) / len(no_call), 3)
    summary["avg_tool_calls"] = round(sum(len(r["calls"]) for r in should_call) / len(should_call), 2)
    summary["avg_seconds"] = round(sum(r["seconds"] for r in records) / len(records), 2)
    print("OVERALL  " + "  ".join(f"{k}={v}" for k, v in summary.items()))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=None, help="default: the configured chat model")
    args = parser.parse_args()

    from diary_agent.core.config import PROJECT_ROOT
    from diary_agent.core.llm import get_embeddings
    from diary_agent.core.prompts import PROMPT_VERSION

    from diary_agent.core.config import settings

    args.models = args.models or [settings.chat_model]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Chroma may hold file locks
        conn, store = seed(Path(tmp), get_embeddings())
        ranks = retrieval_check(store)
        hits = sum(r is not None for r in ranks.values())
        print(f"retrieval (gold rank in top-k, model-free): {ranks}  recall={hits}/{len(ranks)}")

        # The commit pins code changes (like the tool gate) that PROMPT_VERSION doesn't capture.
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        result = {"prompt_version": PROMPT_VERSION, "commit": commit + ("+dirty" if dirty else ""),
                  "today": TODAY.isoformat(), "runs": args.runs,
                  "retrieval_ranks": ranks, "models": {}, "records": []}
        for model_name in args.models:  # one model at a time: both don't fit in 8 GB VRAM together
            records = run_model(model_name, args.runs, conn, store)
            result["models"][model_name] = report(model_name, records, args.runs)
            result["records"] += records
        conn.close()

    out_dir = PROJECT_ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tool_probe-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"\nsaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
