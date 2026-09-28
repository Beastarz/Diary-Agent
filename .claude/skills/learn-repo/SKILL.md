---
name: learn-repo
description: Teach the user how this diary-agent repo works and what AI-agent concepts it demonstrates, using a persistent knowledge file (LEARNINGS.md) instead of rescanning the codebase. Use this skill whenever the user wants to understand or learn from the repo, e.g. "explain how extraction works", "what does db.py do", "what did I learn in M2", "give me learning points", "quiz me", "walk me through the flow", "where is X handled", "why did we do Y". Also use it after a milestone, commit, or significant code change to update the learnings ("update learnings", "sync the notes", "I just finished M3"), even if the user doesn't name the skill.
---

# Learn the repo

The user is learning AI agents (LangChain, LangGraph, tools, memory, evals) by building this diary agent milestone by milestone. This skill turns the repo into a tutor. It explains concepts in terms of *this* code, and it keeps its knowledge in `LEARNINGS.md` (next to this file) so each session starts from what's already known, not from a full rescan.

## Files and who owns what

| File | Owner | Purpose |
|---|---|---|
| `LEARNINGS.md` (this skill's folder) | you (the AI) | Code map, concept index, decision log, learner progress. Your memory of the repo. |
| `docs/NOTES.md` | the user | Per-milestone narrative: experiments, eval tables, "try this". Read it, link to it, but don't rewrite it unless asked. |
| `docs/PLAN.md` | the user | Roadmap and milestone status. Source of truth for what's done and what's next. |

Don't copy NOTES.md into LEARNINGS.md. Point to it ("see NOTES.md → M2 → eval loop") and keep LEARNINGS.md about *where things live in the code and why*.

## Step 1: Always start from LEARNINGS.md

Read `LEARNINGS.md` first. Then check whether it's stale:

```
git rev-parse --short HEAD
git diff --name-only <last_synced_commit>          # committed + uncommitted changes to tracked files
git ls-files --others --exclude-standard            # new untracked files
```

(`<last_synced_commit>` is in the file's header.) These commands work in both PowerShell and bash.

- **No relevant changes:** answer from LEARNINGS.md. Open a source file only when the answer needs actual code (a snippet, an exact signature, a subtle detail). Then open just that file, or a targeted Grep, never a sweep.
- **Changed files:** read only those files (and `git diff <last_synced_commit> -- <file>` if the full file is big). Update LEARNINGS.md (Step 3) before or while answering, so the next session benefits.
- **Missing or unreadable `last_synced_commit`** (e.g. history rewritten): fall back to reading only the files listed in the File map plus anything new from `git ls-files src scripts tests evals`, then resync.

Ignore changes to `uv.lock`, `.gitignore`, `data/`, and caches. They don't carry learning value.

## Step 2: Do what the user asked

### "Explain X" / "How does X work?" / "Where is X?"
Answer in terms of this repo: name the file and the symbol (`extract.py → apply_guardrails()`), then explain the general concept it illustrates, then the *why* (the design decision or the bug that motivated it). The user wants transferable understanding. "This is how LangChain structured output works, and here's the gotcha we hit" beats a line-by-line walkthrough.

Use symbol names rather than line numbers when writing into LEARNINGS.md, because line numbers go stale with every edit. In a chat answer, `file:line` is fine because it's clickable right now.

### "Give me learning points" / "What did I learn in M_n?"
Produce concise, numbered points. Each has:
1. **Concept** in plain words (one line).
2. **In this repo:** file → symbol.
3. **Why it matters / gotcha:** the lesson, ideally one the user actually hit (these stick best).

Order them from foundational to advanced. Aim for 5–8 points unless asked otherwise. Quality beats coverage.

### "Quiz me"
Ask 3–5 questions, one at a time if the user seems to want interaction, otherwise as a list with answers hidden behind a "reply when ready" prompt. Mix recall ("which schema adds entry_date?"), reasoning ("why must every field be required for Ollama?"), and prediction ("what breaks if you remove `max_length` from tags?"). Favor areas marked shaky in **Learner progress**. Afterward, record results there.

### "Walk me through the flow"
Use the **End-to-end flow** section as the spine and expand each hop with the file and function involved.

### "Update learnings" / after a milestone or commit
Run the sync in Step 3 and report briefly what changed in the file.

## Step 3: Keep LEARNINGS.md current

Update it whenever you learned something about the repo that the file doesn't capture yet: after a sync, after answering a question that needed code the file lacked, or after the user corrects your understanding. Keep edits surgical:

- Update the **File map** entries for changed or new files. Remove entries for deleted files.
- Add new concepts to the **Concept index** under the milestone where they appeared.
- Add to the **Decision log** when code or a commit message reveals a "we chose A over B because…".
- Update the **End-to-end flow** if the architecture changed (M3 replaces the chat loop with a LangGraph `StateGraph`, for example).
- Move items out of **Open questions / coming next** once they're answered.
- Set `last_synced_commit` to the current short HEAD and `last_synced` to today's date. If uncommitted changes were included, write `(+ uncommitted)` after the SHA. The next diff against that SHA will still pick them up, which is harmless.

Keep the file skimmable. It's loaded every session, so every line costs attention. Aim for under ~250 lines. When it grows past that, compress older milestones into tighter summaries rather than dropping them.

### Learner progress
This section records what the user has shown they understand, what confused them, and quiz results. Update it when the evidence is clear (they explained something back correctly, got a quiz question wrong, asked the same thing twice). Use it to pitch explanations: skip basics they've mastered and revisit shaky spots with a fresh angle.
