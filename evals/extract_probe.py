"""M2 mini eval: how good is entry extraction on the real model?

Run: uv run python evals/extract_probe.py [--runs 3] [--model llama3.1:8b]

Each case has a transcript and expectations. Checks are deterministic (no LLM judge yet):
- valid:          extraction succeeded
- mood:           inside the acceptable range [lo, hi] (null accepted only if null_ok).
                  Ranges, not exact scores: mood is subjective, and an eval that demands "6"
                  when 5-7 is equally defensible measures noise, not quality.
- no_invented:    fields expected to be empty really are empty
- mentions:       expected keywords appear somewhere in highlights/challenges/gratitude
"""

import argparse
import os
from collections import defaultdict

CASES = [
    {
        "id": "good_day_explicit",
        "transcript": "User: Honestly a great day, I'd say 8 out of 10. Got promoted at work!\n"
        "Assistant: Congratulations! How did you celebrate?\n"
        "User: Took my partner out for sushi. I'm really grateful for my team.",
        "mood": (7, 9),
        "mentions": ["promot", "sushi", "team"],
        "empty": [],
    },
    {
        "id": "bad_day_implicit",
        "transcript": "User: Terrible day. Missed my bus, spilled coffee on my laptop, and my boss yelled at me.\n"
        "Assistant: That sounds really rough. How are you feeling now?\n"
        "User: Exhausted and kind of defeated.",
        "mood": (1, 3),
        "mentions": ["bus", "coffee", "boss"],
        "empty": ["gratitude"],
    },
    {
        "id": "no_mood_stated",
        "transcript": "User: Went to the dentist in the morning, then did groceries and cleaned the kitchen.\n"
        "Assistant: A productive day of errands! Anything else?\n"
        "User: Nope, that's about it.",
        # v1 label demanded null, but the schema says null only when there are no events
        # or feelings at all. Errands with no emotion = neutral, so 4-7 or null are both fine.
        "mood": (4, 7),
        "null_ok": True,
        "mentions": ["dentist", "groceries"],
        "empty": ["gratitude", "mood_words"],
    },
    {
        "id": "mixed_day",
        "transcript": "User: Long day at work, but dinner with my sister was great.\n"
        "Assistant: What made the dinner special?\n"
        "User: We tried a new ramen place and just laughed a lot. Work was draining though.",
        "mood": (5, 7),
        "mentions": ["sister", "ramen", "work"],
        "empty": ["gratitude"],
    },
    {
        "id": "assistant_not_facts",
        "transcript": "User: Pretty quiet day, read a book.\n"
        "Assistant: Nice! Did you also go for a run or see friends today?\n"
        "User: No, just reading. Felt calm.",
        "mood": (5, 8),  # "quiet day, felt calm" - neutral to good
        "mentions": ["book"],
        "empty": ["challenges", "gratitude"],
        "forbidden": ["run", "friend"],  # suggested by the assistant, never happened
    },
    {
        # Found in a real CLI run: the assistant's word "invigorating" became the user's mood.
        "id": "assistant_embellishes",
        "transcript": "User: I went for a run.\n"
        "Assistant: You got out and moved your body today, which can be really invigorating. "
        "What did you enjoy most about it?",
        "mood": (4, 8),
        "null_ok": True,
        "mentions": ["run"],
        "empty": ["mood_words", "challenges", "gratitude"],
        "forbidden": ["invigorat"],
    },
    {
        "id": "gratitude_only",
        "transcript": "User: I just want to note I'm thankful my mom's surgery went well.\n"
        "Assistant: That's wonderful news. How are you feeling?\n"
        "User: Relieved. So relieved.",
        "mood": (7, 10),
        "mentions": ["mom", "surgery"],
        "empty": [],
    },
]


def check(case, entry) -> dict[str, bool]:
    text = " ".join(entry.highlights + entry.challenges + entry.gratitude + [entry.summary]).lower()
    results = {"valid": True}
    lo, hi = case["mood"]
    if entry.mood is None:
        results["mood"] = case.get("null_ok", False)
    else:
        results["mood"] = lo <= entry.mood <= hi
    results["no_invented"] = all(not getattr(entry, f) for f in case["empty"]) and not any(
        w in text for w in case.get("forbidden", [])
    )
    results["mentions"] = all(k in text for k in case["mentions"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if args.model:
        os.environ["DIARY_CHAT_MODEL"] = args.model

    # Imported after setting the env var so config picks it up.
    from diary_agent.core.config import settings
    from diary_agent.journal.extract import ExtractionError, build_extractor, extract_entry, get_extraction_model

    extractor = build_extractor(get_extraction_model())
    totals: dict[str, list[bool]] = defaultdict(list)
    print(f"model={settings.chat_model} runs={args.runs}\n")
    for case in CASES:
        row = defaultdict(int)
        for _ in range(args.runs):
            try:
                entry = extract_entry(extractor, case["transcript"])
                res = check(case, entry)
            except ExtractionError:
                entry, res = None, {"valid": False, "mood": False, "no_invented": False, "mentions": False}
            for k, v in res.items():
                row[k] += v
                totals[k].append(v)
        cells = "  ".join(f"{k}={row[k]}/{args.runs}" for k in ("valid", "mood", "no_invented", "mentions"))
        print(f"{case['id']:<22} {cells}")
        if entry:  # show the last sample so you can eyeball failures
            print(f"{'':<22} -> mood={entry.mood} words={entry.mood_words} hl={entry.highlights} ch={entry.challenges} "
                  f"gr={entry.gratitude} tags={entry.tags}")
    print("\nOVERALL  " + "  ".join(f"{k}={sum(v) / len(v):.0%}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
