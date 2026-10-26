"""Entries in a date range -> SummaryRecord, map-reduce style.

    entries --map: day_line per entry (code, no LLM)----------------> digests
            --collapse: merge groups while still too long (LLM)----> fewer digests
            --reduce: digests + computed stats -> PeriodSummary (LLM, structured)

Why not paste a month of transcripts into one prompt? Context budgeting. Ollama's default window
is a few thousand tokens and anything past it is cut off *silently*: the model just never sees
the start of the month. A day's transcript can be thousands of tokens; its day_line is ~60.

The map step used to be an LLM call per entry that condensed the transcript. It was dropped:
extraction has already condensed each day into a summary + highlights + challenges, and the user
approved those when saving. Re-condensing the transcript cost one call per entry, was never
reviewed by anyone, and was where a distortion crept in ("started a pottery class" ->
"completed"). Now the only LLM calls are reduce, plus collapse when over budget.

The numbers (mood average, best/worst day, tag counts) are computed in code and handed to the
reduce prompt. The LLM only writes the words.
"""

from collections import Counter
from datetime import date, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from diary_agent.core.llm import get_chat_model
from diary_agent.core.prompts import COLLAPSE_SYSTEM, PROMPT_VERSION, REDUCE_SYSTEM
from diary_agent.models.entry import JournalEntry
from diary_agent.models.summary import Period, PeriodSummary, SummaryRecord
from diary_agent.services.extraction import field_guide, invoke_structured

# The budget, in tokens. One fixed window for every summary call: changing num_ctx between
# requests makes Ollama reload the model, which costs seconds. 8k fits in 8 GB VRAM with llama3.1:8b.
NUM_CTX = 8192
REDUCE_INPUT_TOKENS = 3000  # digests per reduce/collapse prompt; leaves room for instructions + output
MAX_COLLAPSE_ROUNDS = 3  # a model that won't shorten text must not loop forever
MIN_TREND_DAYS = 3  # fewer rated days than this: no LLM-written mood trend
CHARS_PER_TOKEN = 4  # rough rule for English. Good enough for a budget; not a tokenizer.


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def get_summary_model(max_tokens: int) -> BaseChatModel:
    # temperature 0: a review should be a faithful reading, not a creative one.
    return get_chat_model(temperature=0, max_tokens=max_tokens, num_ctx=NUM_CTX)


def build_writer(model: BaseChatModel) -> Runnable:
    """messages -> plain text. Used for the collapse step."""
    return model | StrOutputParser()


def build_reducer(model: BaseChatModel) -> Runnable:
    return model.with_structured_output(PeriodSummary, method="json_schema", include_raw=True)


def period_range(period: Period, day: date) -> tuple[date, date]:
    """The calendar week (Mon-Sun) or month containing `day`."""
    if period == "week":
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=6)
    start = day.replace(day=1)
    next_month = (start + timedelta(days=32)).replace(day=1)
    return start, next_month - timedelta(days=1)


def mood_avg(entries: list[JournalEntry]) -> float | None:
    moods = [e.mood for e in entries if e.mood is not None]
    return round(sum(moods) / len(moods), 1) if moods else None


def stats_text(entries: list[JournalEntry], start: date, end: date) -> str:
    """Facts for the reduce prompt. 8B models miscount and misaverage, so code does it.
    Not every computable fact helps: a "Mood trend: down" line made llama narrate the direction
    and drop the low/high days (named in 2/6 runs vs 6/6 without it; evals/summary_probe.py)."""
    rated = [e for e in entries if e.mood is not None]
    # Coverage: with 1 entry in a month, llama wrote "a consistently good month".
    lines = [f"Entries: {len(entries)} of the {(end - start).days + 1} days in this period"]
    if rated:
        low = min(rated, key=lambda e: e.mood)
        high = max(rated, key=lambda e: e.mood)
        lines += [
            "Mood by day: " + ", ".join(f"{e.entry_date:%a %d} {e.mood}" for e in rated),
            f"Mood average: {mood_avg(entries)}/10, lowest {low.mood} on {low.entry_date:%A %d}, "
            f"highest {high.mood} on {high.entry_date:%A %d}",
        ]
    tags = Counter(t for e in entries for t in set(e.tags))
    recurring = [f"{t} ({n} days)" for t, n in tags.most_common() if n >= 2]
    if recurring:
        lines.append("Tags on 2+ days: " + ", ".join(recurring))
    return "\n".join(lines)


def day_line(entry: JournalEntry) -> str:
    """The map step, in code: one day for the reduce prompt, built only from fields the user
    approved when saving. The transcript is never read here."""
    mood = f"mood {entry.mood or '?'}"
    if entry.mood_words:
        mood += f", felt {', '.join(entry.mood_words)}"
    parts = [f"{entry.entry_date:%a %Y-%m-%d} ({mood}): {entry.summary}"]
    for label, items in (("Highlights", entry.highlights), ("Challenges", entry.challenges),
                         ("Grateful for", entry.gratitude)):
        if items:
            parts.append(f"{label}: {'; '.join(items)}.")
    return " ".join(parts)


def _groups(digests: list[str], budget: int) -> list[list[str]]:
    """Consecutive digests packed greedily so each group fits one prompt. Keeps time order."""
    groups: list[list[str]] = [[]]
    for d in digests:
        if groups[-1] and estimate_tokens("\n\n".join([*groups[-1], d])) > budget:
            groups.append([])
        groups[-1].append(d)
    return groups


def collapse(writer: Runnable, digests: list[str], budget: int = REDUCE_INPUT_TOKENS) -> list[str]:
    """Only runs when the digests are still over budget (rare: 31 days x ~60 tokens fits).
    Merging loses detail, so it's a last resort, not a default."""
    for _ in range(MAX_COLLAPSE_ROUNDS):
        if len(digests) <= 1 or estimate_tokens("\n\n".join(digests)) <= budget:
            break
        prompts = [[SystemMessage(COLLAPSE_SYSTEM), HumanMessage("\n\n".join(g))] for g in _groups(digests, budget)]
        digests = [d.strip() for d in writer.batch(prompts)]
    return digests


def reduce(reducer: Runnable, digests: list[str], stats: str, period: Period, max_retries: int = 2) -> PeriodSummary:
    messages: list[BaseMessage] = [
        SystemMessage(f"{REDUCE_SYSTEM.format(period=period)}\n\nFields:\n{field_guide(PeriodSummary)}"),
        HumanMessage(f"{stats}\n\nDigests:\n\n" + "\n\n".join(digests) + f"\n\nWrite the {period} review."),
    ]
    return invoke_structured(reducer, messages, max_retries, f"{period} summary")


def apply_guardrails(summary: PeriodSummary, entries: list[JournalEntry]) -> PeriodSummary:
    """Rules the model still broke in evals/summary_probe.py, enforced in code (like extraction's)."""
    updates: dict = {}
    # "good mood" kept turning up as a win, even with the field description saying it isn't one.
    for field in ("wins", "challenges"):
        items = getattr(summary, field)
        kept = [i for i in items if "mood" not in i.lower()]
        if kept != items:
            updates[field] = kept
    # Two points aren't a trend, but llama wrote "remained steady throughout the period" for
    # 2 days out of 7 in 3/3 runs, even with a prompt rule against it. Code writes it instead.
    rated = [e for e in entries if e.mood is not None]
    if len(rated) < MIN_TREND_DAYS:
        days = ", ".join(f"{e.entry_date:%A %d} {e.mood}/10" for e in rated) or "none"
        updates["mood_trend"] = f"Too few rated days for a trend ({days})."
    return summary.model_copy(update=updates) if updates else summary


def summarize_period(
    writer: Runnable,
    reducer: Runnable,
    entries: list[JournalEntry],
    period: Period,
    start: date,
    end: date,
    model_name: str,
) -> SummaryRecord | None:
    """None when there's nothing to summarize: no LLM call for an empty week."""
    if not entries:
        return None
    entries = sorted(entries, key=lambda e: e.entry_date)  # the repository returns newest first
    digests = collapse(writer, [day_line(e) for e in entries])
    summary = apply_guardrails(reduce(reducer, digests, stats_text(entries, start, end), period), entries)
    return SummaryRecord(
        **summary.model_dump(),
        period=period,
        start_date=start,
        end_date=end,
        entry_count=len(entries),
        mood_avg=mood_avg(entries),
        model=model_name,
        prompt_version=PROMPT_VERSION,
    )
