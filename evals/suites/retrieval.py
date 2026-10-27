"""Retrieval suite: embeddings only, no chat model. Deterministic, so runs is ignored.

Seeds a temp Chroma index with retrieval_journal.jsonl (real nomic-embed-text vectors, same
upsert_entry as the app) and asks retrieval_questions.jsonl. Two layers, as search_entries uses them:
- ranking:  is a gold day in the raw top-k?  (recall@1, recall@SEARCH_K, MRR over the top 10)
- cutoff:   after MAX_DISTANCE, does the tool keep the answer, and drop everything for "never
            happened" questions? This is what the model actually receives.
The distance stats show where the cutoff sits between relevant and unrelated matches.
"""

import tempfile
from datetime import date
from pathlib import Path

from evaluators import mrr, rank_of, rate, recall_at
from suites import load_jsonl, seed_journal

DEPTH = 10  # how deep to rank for MRR; the tool itself only looks at SEARCH_K


def run(runs: int) -> dict:
    from diary_agent.repositories import vector_repository as vector_repo
    from diary_agent.services.agent.tools import MAX_DISTANCE, SEARCH_K

    cases, failures = [], []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Chroma may hold file locks
        conn, store = seed_journal("retrieval_journal.jsonl", Path(tmp))
        conn.close()  # only the index is searched here
        for q in load_jsonl("retrieval_questions.jsonl"):
            hits = vector_repo.search(store, q["q"], k=DEPTH)
            gold = {date.fromisoformat(d) for d in q["gold"]}
            ranked = [d for d, _ in hits]
            kept = [d for d, dist in hits[:SEARCH_K] if dist <= MAX_DISTANCE]  # what search_entries returns
            record = {"id": q["id"], "q": q["q"], "gold": q["gold"], "answerable": bool(gold),
                      "top": [(d.isoformat(), round(dist, 3)) for d, dist in hits[:SEARCH_K]],
                      "kept": [d.isoformat() for d in kept]}
            if gold:
                record["rank"] = rank_of(gold, ranked)
                record["gold_distance"] = next((round(dist, 3) for d, dist in hits if d in gold), None)
                record["tool_ok"] = any(d in gold for d in kept)
            else:
                record["top_distance"] = round(hits[0][1], 3)
                record["tool_ok"] = not kept
            cases.append(record)
            if not record["tool_ok"]:
                if gold:
                    why = (f"rank {record['rank']}, distance {record['gold_distance']}" if record["rank"]
                           else f"not in top {DEPTH}")
                else:
                    why = f"kept {record['kept']} (closest {record['top_distance']})"
                failures.append(f"{q['id']:<16} {q['q']!r}: {why}")

    answerable = [c for c in cases if c["answerable"]]
    unanswerable = [c for c in cases if not c["answerable"]]
    ranks = [c["rank"] for c in answerable]
    gold_d = [c["gold_distance"] for c in answerable if c["gold_distance"] is not None]
    metrics = {
        "recall_at_1": recall_at(ranks, 1),
        f"recall_at_{SEARCH_K}": recall_at(ranks, SEARCH_K),
        "mrr": mrr(ranks),
        "tool_keeps_answer": rate([c["tool_ok"] for c in answerable]),
        "tool_rejects_never_happened": rate([c["tool_ok"] for c in unanswerable]),
        "max_gold_distance": max(gold_d) if gold_d else 0.0,
        "min_unanswerable_distance": min(c["top_distance"] for c in unanswerable) if unanswerable else 0.0,
        "n_questions": len(cases),
    }
    return {"metrics": metrics, "cases": cases, "failures": failures,
            "config": {"search_k": SEARCH_K, "max_distance": MAX_DISTANCE}}
