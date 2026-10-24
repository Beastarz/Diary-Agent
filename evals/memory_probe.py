"""M6 probe: does the memory write path keep lasting facts, drop one-day noise, and dedupe?

Run: uv run python evals/memory_probe.py [--runs 2] [--part extract|similarity|tool]

Three parts, all against the real Ollama models:
- extract:    transcripts -> extract_memories(). Per case: did each expected fact come out
              (a keyword in some memory), and did anything come out that shouldn't (pollution)?
              Raw = the model's list before guardrails, kept = after.
- similarity: nomic-embed-text cosine for pairs that should be duplicates, updates, or
              different memories. Calibrates DUPLICATE_SCORE / RELATED_SCORE in services/memory.py.
- tool:       which tool llama3.1 calls for "remember ..." requests vs look-alikes.
"""

import argparse
import math
import time
from datetime import date
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from diary_agent.core.llm import get_chat_model, get_embeddings
from diary_agent.core.prompts import TOOLS_SYSTEM
from diary_agent.services import memory as memory_service
from diary_agent.services.agent.nodes import _companion_prompt, calendar_hint, tools_for
from diary_agent.services.extraction import get_extraction_model

# expect: one keyword list per memory that should come out (any keyword matching counts).
CASES = [
    {"id": "goal", "expect": [["marathon"]], "transcript": [
        "Went for a long run before work, 18k.",
        "I'm training for the Berlin marathon in March, so the long runs are every Sunday now."]},
    {"id": "person", "expect": [["mia", "sister"]], "transcript": [
        "Had dinner with my sister Mia. She just moved back to Leeds.",
        "It was nice, we talked for hours."]},
    {"id": "job_and_pet", "expect": [["nurse", "hospital"], ["dog", "biscuit"]], "transcript": [
        "Long shift at the hospital, I've been a nurse there for six years.",
        "Came home and took my dog Biscuit for a walk, which helped."]},
    {"id": "preference", "expect": [["morning", "mornings"]], "transcript": [
        "I always journal better in the mornings, evenings I'm too tired to think.",
        "Today was fine though."]},
    # Nothing lasting: a mood, a one-off event, a plan for tomorrow.
    {"id": "only_today", "expect": [], "transcript": [
        "Pretty tired today, the meeting ran late.",
        "Tomorrow I need to buy groceries and call the bank."]},
    {"id": "small_talk", "expect": [], "transcript": ["hi", "not much, just wanted to say hello"]},
    # The assistant suggests things; none of it is about the user.
    {"id": "assistant_words", "expect": [], "assistant": "Maybe yoga or meditation could help you unwind?",
     "transcript": ["Work was stressful.", "Maybe, I don't know."]},
    # Mixed: one lasting fact buried in a day's events.
    {"id": "mixed", "expect": [["spanish"]], "transcript": [
        "Rainy day. Missed the bus and was late.",
        "At least my Spanish class went well, I've been learning for a year now."]},
]

PAIRS = {
    "duplicate": [
        ("Training for a marathon", "Training for the Berlin marathon"),
        ("Sister named Mia", "Has a sister called Mia"),
        ("Works as a nurse", "Is a nurse at a hospital"),
    ],
    "update": [
        ("Training for a marathon in March", "Training for a half marathon in May"),
        ("Lives in London", "Moved to Leeds"),
        ("Works as a nurse", "Now works as a midwife"),
    ],
    "different": [
        ("Sister named Mia", "Brother named Tom"),
        ("Training for a marathon", "Learning Spanish"),
        ("Has a dog called Biscuit", "Prefers journaling in the morning"),
    ],
}

TOOL_CASES = [
    ("Remember that I'm vegetarian.", "remember_fact"),
    ("Please remember my sister's name is Mia.", "remember_fact"),
    ("Don't forget I'm training for a marathon in March.", "remember_fact"),
    ("Remember when I ran the 10k? How did that go?", "search_entries"),
    ("I remember the beach trip with my dad, it was lovely.", None),
]


def transcript(case: dict) -> str:
    lines = []
    for i, text in enumerate(case["transcript"]):
        lines.append(f"User: {text}")
        if i == 0:
            lines.append(f"Assistant: {case.get('assistant', 'That sounds like a lot. How did it feel?')}")
    return "\n".join(lines)


def score(expect: list[list[str]], contents: list[str]) -> tuple[int, int]:
    """(expected facts found, unexpected memories)."""
    low = [c.lower() for c in contents]
    found = sum(any(k in c for c in low for k in keys) for keys in expect)
    extra = sum(not any(k in c for keys in expect for k in keys) for c in low)
    return found, extra


def run_extract(runs: int) -> None:
    extractor = memory_service.build_memory_extractor(get_extraction_model())
    totals = {"expected": 0, "found": 0, "raw_extra": 0, "kept_extra": 0, "dropped": 0}
    for case in CASES:
        text = transcript(case)
        for r in range(runs):
            started = time.perf_counter()
            raw = memory_service.extract_candidates(extractor, text)
            kept = memory_service.apply_memory_guardrails(raw, text)
            found, kept_extra = score(case["expect"], [m.content for m in kept])
            _, raw_extra = score(case["expect"], [m.content for m in raw])
            totals["expected"] += len(case["expect"])
            totals["found"] += found
            totals["raw_extra"] += raw_extra
            totals["kept_extra"] += kept_extra
            totals["dropped"] += len(raw) - len(kept)
            ok = found == len(case["expect"]) and kept_extra == 0
            print(f"[{'ok ' if ok else 'BAD'}] {case['id']:16} run {r + 1}: found {found}/{len(case['expect'])}, "
                  f"extra {kept_extra} (raw {raw_extra}) {time.perf_counter() - started:.1f}s")
            for m in raw:
                mark = "kept" if m in kept else "drop"
                print(f"        {mark} ({m.kind}) {m.content!r}  <- {m.evidence!r}")
    print(f"\nrecall {totals['found']}/{totals['expected']}, pollution kept {totals['kept_extra']} "
          f"(raw {totals['raw_extra']}), guardrails dropped {totals['dropped']}")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def run_similarity() -> None:
    emb = get_embeddings()
    print(f"thresholds: duplicate >= {memory_service.DUPLICATE_SCORE}, update >= {memory_service.RELATED_SCORE}")
    for label, pairs in PAIRS.items():
        for a, b in pairs:
            va, vb = emb.embed_documents([a])[0], emb.embed_query(b)  # stored vs query, as in propose()
            s = cosine(va, vb)
            verdict = ("duplicate" if s >= memory_service.DUPLICATE_SCORE
                       else "update" if s >= memory_service.RELATED_SCORE else "different")
            # Rewording treated as an update is harmless (the user confirms old -> new); a missed
            # update leaves two memories; a different fact treated as an update loses data.
            ok = verdict == label or (label == "duplicate" and verdict == "update")
            bad = "LOSS" if label == "different" and verdict != "different" else "miss"
            print(f"[{'ok  ' if ok else bad}] {label:9} {s:.3f} -> {verdict:9} {a!r} / {b!r}")


def run_tool(runs: int) -> None:
    model = get_chat_model()
    day = date.today()
    ctx = SimpleNamespace(day=day)
    hits = 0
    for text, want in TOOL_CASES:
        for r in range(runs):
            state = {"user_input": text, "messages": [HumanMessage(text)]}
            runtime = SimpleNamespace(context=ctx, store=None)
            prompt = _companion_prompt(state, runtime, TOOLS_SYSTEM.format(calendar=calendar_hint(day)))
            offered = tools_for(text)  # the same code gate as chat_agent
            reply = model.bind_tools(offered).invoke(prompt) if offered else model.invoke(prompt)
            got = reply.tool_calls[0]["name"] if reply.tool_calls else None
            args = reply.tool_calls[0]["args"] if reply.tool_calls else ""
            hits += got == want
            print(f"[{'ok ' if got == want else 'BAD'}] {text!r} run {r + 1}: {got} {args}")
    print(f"\ncorrect tool {hits}/{len(TOOL_CASES) * runs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--part", choices=["extract", "similarity", "tool"], default=None, help="default: all")
    args = parser.parse_args()
    if args.part in (None, "similarity"):
        run_similarity()
        print()
    if args.part in (None, "extract"):
        run_extract(args.runs)
        print()
    if args.part in (None, "tool"):
        run_tool(args.runs)


if __name__ == "__main__":
    main()
