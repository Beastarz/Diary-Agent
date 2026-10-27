"""Scoring functions for the eval harness.

Pure functions: no models, no I/O. They're unit-tested (evals/tests/), so when a score drops
you can trust it's the agent that got worse, not the checker. M4 and M5 both had "failures"
that were really the checker missing a correct answer; testing the checker is the fix.
"""

import re
from collections import defaultdict
from datetime import date


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


# --- language ----------------------------------------------------------------------------

# Japanese kana, CJK ideographs, Hangul. qwen3 leaked these into English replies in M1.
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def has_cjk(text: str) -> bool:
    return bool(CJK.search(text))


# --- routing -----------------------------------------------------------------------------

def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """expected -> predicted -> count. Off-diagonal cells are the mistakes, and which way they go."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for expected, predicted in pairs:
        table[expected][predicted] += 1
    return {e: dict(p) for e, p in table.items()}


# --- retrieval ---------------------------------------------------------------------------

def rank_of(gold: set[date], ranked: list[date]) -> int | None:
    """1-based rank of the first gold day in the results, or None if no gold day came back."""
    for i, day in enumerate(ranked, start=1):
        if day in gold:
            return i
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    """Share of questions whose answer is in the top k."""
    return rate([r is not None and r <= k for r in ranks])


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank: 1.0 if the answer is always first, 0.5 if always second, 0 if never found."""
    return sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0


# --- extraction --------------------------------------------------------------------------

def matches(text: str, pattern: str) -> bool:
    """Case-insensitive substring match. "a|b" means either. Substrings, so "promot" matches
    "promoted" and "promotion": keywords are stems, not whole words."""
    text = text.lower()
    return any(alt.strip().lower() in text for alt in pattern.split("|"))


def check_extraction(case: dict, entry) -> dict[str, bool]:
    """One extracted entry vs a case's expectations. A check is only scored if the case defines it,
    so every case tests what it was written for and nothing else.

    Case keys (all optional except mood):
      mood: [lo, hi]     null_ok: bool          a range, because 6 vs 7 is opinion, not error
      mentions: [...]    each pattern appears somewhere in highlights/challenges/gratitude/summary
      placement: {field: [...]}   each pattern appears in that specific list
      not_in: {field: [...]}      no pattern appears in that list ("8 out of 10" isn't a highlight)
      empty: [fields]    these lists are empty
      forbidden: [...]   no pattern appears in ANY field, tags and mood words included
      tags_any: [...]    at least one tag matches;   tags_not: [...]  no tag matches
      mood_words_any: [...]
    """
    def field_text(field: str) -> str:
        value = getattr(entry, field)
        return " ; ".join(value) if isinstance(value, list) else str(value)

    content = " ; ".join(field_text(f) for f in ("highlights", "challenges", "gratitude", "summary"))
    everything = " ; ".join([content, field_text("mood_words"), field_text("tags")])
    res: dict[str, bool] = {}

    lo, hi = case["mood"]
    res["mood"] = case.get("null_ok", False) if entry.mood is None else lo <= entry.mood <= hi
    if "mentions" in case:
        res["mentions"] = all(matches(content, p) for p in case["mentions"])
    if "placement" in case:
        res["placement"] = all(matches(field_text(f), p) for f, ps in case["placement"].items() for p in ps)
    if "not_in" in case:
        res["not_in"] = not any(matches(field_text(f), p) for f, ps in case["not_in"].items() for p in ps)
    if "empty" in case:
        res["empty"] = all(not getattr(entry, f) for f in case["empty"])
    if "forbidden" in case:
        res["forbidden"] = not any(matches(everything, p) for p in case["forbidden"])
    if "tags_any" in case or "tags_not" in case:
        tags = field_text("tags")
        res["tags"] = (not case.get("tags_any") or any(matches(tags, p) for p in case["tags_any"])) and not any(
            matches(tags, p) for p in case.get("tags_not", [])
        )
    if "mood_words_any" in case:
        res["mood_words"] = any(matches(field_text("mood_words"), p) for p in case["mood_words_any"])
    return res


# --- tool use ----------------------------------------------------------------------------

# Ways a reply admits nothing was found. Keyword lists miss paraphrases; the M7 judge replaces this.
ADMITS = ("couldn't", "could not", "can't find", "cannot find", "didn't find", "no entries", "no entry",
          "nothing", "don't see", "do not see", "no record", "not find", "no mention", "doesn't look like",
          "don't have", "no matching", "don't mention", "doesn't mention", "not mention", "haven't",
          "no sign", "isn't any", "there's no")


def covers(call: dict, case: dict) -> bool:
    """Does one date-tool call's range include every "cover" day (or any "cover_any" day) and
    none of the "exclude" days? A search with no dates never counts as covering."""
    args = call["args"]
    try:
        start = date.fromisoformat(args["start_date"]) if args.get("start_date") else None
        end = date.fromisoformat(args["end_date"]) if args.get("end_date") else None
    except (ValueError, TypeError):
        return False
    if start is None and end is None:
        return False
    if call["name"] == "get_entries_by_date":
        end = end or start  # one day when only start_date is given
    lo, hi = start or date.min, end or date.max
    inside = lambda d: lo <= date.fromisoformat(d) <= hi  # noqa: E731
    if "cover" in case and not all(inside(d) for d in case["cover"]):
        return False
    if "cover_any" in case and not any(inside(d) for d in case["cover_any"]):
        return False
    return not any(inside(d) for d in case.get("exclude", []))


def check_tool_turn(case: dict, calls: list[dict], reply: str) -> dict[str, bool]:
    """One user turn through the graph. calls = [{"name", "args"}] in order; reply = the final answer.
    - tool:   called only acceptable tools, at least one (or none, for "tools": [] cases)
    - args:   date cases: some call covers the right days; topic cases: a non-empty search query
    - answer: the reply mentions a fact from the right entry, or admits nothing was found
    - clean:  no CJK characters (the M1 leak)"""
    names = {c["name"] for c in calls}
    allowed = set(case["tools"])
    res = {"tool": bool(names) and names <= allowed if allowed else not names}
    if allowed:
        relevant = [c for c in calls if c["name"] in allowed]
        if "cover" in case or "cover_any" in case:
            res["args"] = any(covers(c, case) for c in relevant)
        else:
            res["args"] = any(c["args"].get("query", "").strip() for c in relevant if c["name"] == "search_entries") \
                or any(c["name"] == "get_entries_by_date" for c in relevant)
    text = reply.lower()
    if "answer" in case:
        res["answer"] = any(k in text for k in case["answer"])
    if case.get("admits"):
        res["answer"] = any(k in text for k in ADMITS)
    res["clean"] = not has_cjk(reply)
    return res


# --- judge -------------------------------------------------------------------------------

def count_questions(text: str) -> int:
    """Questions asked, counted in code: "ONE follow-up question" is a rule code can check, so the
    judge only grades what code can't (is the question open, and about what the user said?)."""
    return len(re.findall(r"\?+", text))


def agreement(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    """(human, judge) pass/fail pairs -> how far the judge can be trusted on one check.

    Plain agreement hides a lazy judge: if 90% of replies pass, a judge that always says "pass"
    agrees 90% of the time and catches nothing. So the fails get their own numbers:
    fail_catch = of the replies a human failed, how many the judge failed too;
    false_alarm = of the replies a human passed, how many the judge failed anyway."""
    human_fails = [judge for human, judge in pairs if not human]
    human_passes = [judge for human, judge in pairs if human]
    return {
        "agree": rate([human == judge for human, judge in pairs]),
        "fail_catch": rate([not judge for judge in human_fails]),
        "false_alarm": rate([not judge for judge in human_passes]),
        "n_graded": len(pairs),
        "n_human_fails": len(human_fails),
    }


# --- simulated user ----------------------------------------------------------------------

# Ways a simulated user slips out of character into the assistant's voice.
ASSISTANT_VOICE = ("as an ai", "how can i help", "how can i assist", "is there anything else", "i'm here to",
                   "journaling companion", "assistant:", "user:", "i'm glad you", "it sounds like you")


def in_character(text: str, max_chars: int = 400) -> bool:
    """A simulated user's turn is only usable if it reads like a user: non-empty, short, and not
    the assistant's voice. A run with an out-of-character turn is dropped, not scored: it would
    measure the simulator, not the agent."""
    lowered = text.strip().lower()
    return bool(lowered) and len(lowered) <= max_chars and not any(p in lowered for p in ASSISTANT_VOICE)


def said_gold(gold: dict, user_text: str) -> tuple[dict, int]:
    """Gold limited to what the simulated user actually said. The script lists the persona's whole
    day, but a simulator forgets things, so "the entry mentions sushi" is only fair if sushi came up.
    Keeps every check that's about *not* writing something (forbidden, not_in, empty) and the mood.
    Returns (fair gold, number of patterns dropped): the dropped count measures the simulator."""
    fair, dropped = dict(gold), 0
    if "mentions" in gold:
        fair["mentions"] = [p for p in gold["mentions"] if matches(user_text, p)]
        dropped += len(gold["mentions"]) - len(fair["mentions"])
    if "placement" in gold:
        fair["placement"] = {f: [p for p in ps if matches(user_text, p)] for f, ps in gold["placement"].items()}
        dropped += sum(len(ps) for ps in gold["placement"].values()) - sum(len(ps) for ps in fair["placement"].values())
    return fair, dropped
