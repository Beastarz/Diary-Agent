# Save-time extractors and relationships — design

Date: 2026-11-15. Status: draft for review. Supersedes `2026-10-27-orchestrator-design.md`.

## Intent

The diary stays **one agent**: one LangGraph graph, one conversation per day, the same check-in and chat. The user wants to grow it into something that also tracks the people in their life, and later things like expenses, without the complexity of an orchestrator and several agents.

The insight behind this design: in this app the domains do not diverge in the *chat*. They diverge at **save time**, when the day's transcript is turned into records. Today that step already runs two extractors in a row, the journal entry and the long-term memory. This design turns that step into a list of pluggable **extractors**. Relationships become the second real extractor. Expenses, when wanted, becomes the third, as a new folder plus one line of registration.

The second change is that the save path **no longer asks for confirmation**. Entry, memories and people are written automatically. The user's longer-term plan is for a deployed backend to run this chain on a schedule at the end of each day with nobody at the keyboard. Scheduling itself is out of scope here; this design only makes the chain callable as a plain service function so a scheduler can call it later.

This is also a learning build. It should teach: the extractor / registry pattern as a lighter alternative to multi-agent, idempotent writes, and what replaces a human gate when you remove one (code guardrails plus evals).

## How it works

### A `/save` turn, before and after

```
Before:  router -> extract -> confirm(interrupt) -> save -> update_memory -> confirm_memory(interrupt) -> END
After:   router -> extract -> save -> run_extractors -> END
```

- **extract** is unchanged: the whole day's transcript goes to the entry extractor, with the existing retries and guardrails. It is the gate: with no human turn or a failed extraction, nothing else runs and the user gets the notice they get today.
- **save** writes the entry and indexes it, as today. The "an entry already exists, overwrite?" question is gone: one entry per day, a later save replaces it.
- **run_extractors** is one new node. It calls the service function `run_all(transcript, day, deps)`, which walks `EXTRACTORS` in order, runs each, writes each one's results, and returns one report per extractor. One extractor failing (bad JSON after retries, Ollama down, a DB error) is caught, reported, and never stops the others or undoes the entry.
- The turn ends with a single notice built from the reports, for example: `Saved. Remembered 2 things. Noted 1 person: Lucas (new).`

The check-in's final step follows the same path. `/save` is the manual trigger; the CLI's "Save today's entry before leaving?" on exit stays as the day-end trigger until a scheduler exists.

The `confirm_memory` interrupt survives in exactly one place: the chat-time `remember_fact` tool. The user typed "remember that ...", so asking "Remember this?" there is cheap and keeps that write gated. `ask_to_remember` in the CLI stays for it. `ask_to_save` goes; the CLI shows the saved entry with the existing `render_entry` panel instead.

### The extractor contract

Every extractor provides these pieces. The exact Python shape (a dataclass of callables or a small ABC) is the builder's choice.

| Piece | Responsibility |
|---|---|
| `name` | Stable id used in reports, notices and tests: `memory`, `relationships`, later `expenses`. |
| `build(model)` | Returns the runnable for this extractor's structured output, built once at startup, not per turn. The schema follows the M2 rules: all fields required, lists bounded, evidence first. |
| `extract(runnable, transcript)` | LLM call with retries plus this extractor's **code guardrails**. Returns validated items. Never writes. |
| `apply(items, day, deps)` | Writes the items **idempotently**: running it twice for the same day and items yields the same database. Returns a report of what changed. |
| `describe(report)` | One short sentence for the notice, e.g. `Noted 1 person: Lucas (new).` or nothing when nothing changed. |

`deps` is a small frozen dataclass handed in by the caller: the journal connection, the long-term store (may be None), the model name and today's date. The graph builds it from `DiaryContext`; a scheduler builds it from scratch. `run_all` is the only thing the graph node calls, and it has no LangGraph imports, so it can be called from a script, a test, or a future scheduler.

The registry is one explicit list in one file:

```python
EXTRACTORS = [memory.EXTRACTOR, relationships.EXTRACTOR]
```

### Relationships

**What the model fills in.** One flat, bounded schema per person mentioned, evidence first, following the pattern that works for memory:

| Field | Meaning |
|---|---|
| `evidence` | The user's exact words the mention comes from. |
| `name` | How the user refers to the person: `Lucas`, `Mum`, `my boss`. |
| `relation` | One of a fixed set: `family`, `partner`, `friend`, `colleague`, `acquaintance`, `other`. |
| `facts` | Up to 3 lasting things about the person in the user's words, e.g. `good at math`. Empty if none. |
| `what_happened` | One sentence: what happened with this person today. |

The wrapper holds at most 5 mentions. Groups ("my team", "the kids") count as one person if the user treats them as one.

**Code guardrails** (the part that replaces the human gate):

- The name must appear in the **user's** turns of the transcript, case-insensitively. A name that only the assistant used, or that appears nowhere, is dropped. This blocks invented people and names the model picked up from its own replies.
- Each fact must be grounded in the evidence quote, using the same word-overlap check the memory guardrails use.
- Mentions of the user themself are dropped.
- Duplicate names within one run are merged into one mention.

**Storage.** Two new tables in `journal.db`, through a new `people_repository`:

- `people`: one row per person. Normalised name as the key (lowercase, leading "my " stripped; the builder settles the exact rule), the display name, relation, `facts` as a JSON list, `first_seen`, `last_seen`, `updated_at`.
- `mentions`: one row per person per day. Primary key `(entry_date, person_id)`, plus `what_happened` and `evidence`.

**Idempotent apply.** For each mention: upsert the person. New facts are appended without duplicates, capped at 10; `relation` is replaced only when the new value is not `other`; `first_seen` is the earliest date, `last_seen` the latest. Then the day's mentions are replaced: delete all mentions for that date, insert the new ones. Running the chain twice on the same day therefore leaves one mention per person and no duplicate facts.

**Reading people back in chat.** A `people_block` goes into the system prompt alongside the memory block: for every stored person whose name appears in the user's message (plain string match, no LLM, no embeddings), one line with relation, facts and last seen date. This keeps plain chat exactly as fast as today. No new tool. Questions like "when did I last see Lucas?" are answered by the block's `last seen`, and `search_entries` still covers the rest.

**CLI.** `diary people list`, `diary people show <name>` (the person and their mentions), `diary people forget <name>`.

### Memory becomes the first registered extractor

`services/memory.py` keeps everything the `remember_fact` tool also uses: guardrails, `propose`, `apply_proposals`, `memory_block`. The new `extractors/memory.py` is a thin adapter around it for the save-time path, with two differences from today:

- No proposals, no interrupt. `propose` is still used, but only to **skip duplicates**: anything it would have offered as an "update" (overwrite an existing memory) is stored as a new memory instead. With nobody watching, an overwrite is the one write `diary memory forget` cannot undo, and M6 found that embedding similarity cannot tell a reworded fact from a different one. Overwrites stay possible only on the confirmed `remember_fact` path.
- Idempotency is **best effort** here, unlike the people tables: dedupe is similarity-based, so a rerun whose wording differs can add a near-duplicate memory. Accepted, and noted in the learning notes; the strict alternative (tagging save-path memories with their day and replacing them on rerun) is not worth a model and repository change now.
- It **drops items of kind `person`**. People now live in the relationships tables; two homes for the same fact would drift. The `MemoryKind` enum keeps `person` so existing stored memories still load and `remember_fact` keeps working unchanged.

### What the user sees

- After a save: the entry panel, then the one-line notice listing what every extractor wrote.
- At the start of `diary chat`: a one-line recap of the most recent processed day, computed from the repositories, no new table: `Last entry: Fri 14 Nov. Remembered 2 things (diary memory list), noted 1 person: Lucas (diary people list).` This is how an unattended wrong write gets noticed within a day.

## Goals

- The save chain is a list of extractors; adding one is a new module plus one registry line, plus its own CLI subcommands and eval dataset. No edits to the graph, the other extractors, or `run_all`.
- Relationships extracted at save time into their own tables, visible in chat through the prompt block and in the CLI.
- No confirmation on the save path. The chain is callable without the graph.
- Entry and people writes are idempotent per day; memory dedupes by similarity.
- A `relationships` eval suite in the M7 harness, so extraction quality is measured before and after prompt changes.
- Unit tests with a fake extractor prove the contract without Ollama.

## Non-goals

- Scheduling. No cron, no Task Scheduler, no background process. Those come with deployment.
- The HTTP API and the web frontend, including browsing past dates by clicking.
- Expenses, or any third extractor. This design only leaves the slot.
- A people tool for the chat agent. The prompt block is enough for now.
- Migrating existing `person` memories into the people tables.
- Automatic re-extraction of a day after the user keeps chatting. `/save` and the exit prompt are the triggers until a scheduler exists.

## Anti-goals

These would make the work a failure even if it runs:

- **Entry extraction gets worse.** The extraction suite must score within run-to-run noise of its baseline.
- **Plain chat gets slower.** Extractors run only on the save path. The people block is string matching.
- **A wrong write that nobody sees.** Every write appears in the save notice and in the next session's recap, with the command that undoes it.
- **Adding expenses needs edits outside its folder and the registry line.** If it does, the contract is wrong.
- **A rerun duplicates data.** Running the chain twice for a day must leave the entry and the people tables unchanged. Memory is best effort (see above).

## Constraints

- Local only: Ollama, `llama3.1:8b`, no cloud calls. Same stack: LangGraph 1.2.x, `SqliteSaver`, `SqliteStore`, Chroma, typer and rich.
- The layer rule holds: `cli/` and `api/` call `services/`, which call `repositories/`, which use `models/`.
- Ollama structured output: all fields required, lists with a maximum length, descriptions delivered through the prompt (`field_guide`), evidence fields first (M2, M6).
- The 8B model is weak at open-ended judgment, so code decides wherever it can: which names count, which facts are grounded, how records merge.
- Prompts versioned in `core/prompts.py`; the relationships prompt gets its own entry and `PROMPT_VERSION` is bumped.
- One conversation thread per day, thread_id = date.

## Approaches

### Where to put the domain split

1. **Orchestrator with agent subgraphs** (the superseded design). A planner splits each message into jobs for separate agents with their own state, prompts and interrupts. Rejected: in this app the domains share one conversation and only diverge at save time, so the planner, job queue and floor-holding were machinery without a job. It also put an open-ended LLM decision on every multi-part message, which the 8B model handles poorly.
2. **Add nodes by hand.** Copy the memory pattern: an `extract_people` node and a confirm node, wired after memory. Smallest diff, but every future domain means copying the pair and editing the graph and the CLI again.
3. **Extractor registry with a generic runner** (chosen). The post-save step becomes a list; the graph has one node that runs the list. Memory moves into the list first, relationships second, so the contract is proven by two real members before a third exists.
4. **One combined extraction call** returning entry, people and memories together. Fewer LLM calls, but large nested schemas fail more on the 8B model (M2), one failure loses everything, and every new domain edits a shared schema. Rejected.

### What a relationship record is

1. **Mentions only** (date, name, what happened). Simple, but "who is Lucas?" means re-reading every mention.
2. **People only** (name, relation, facts). Loses the history of when you saw them.
3. **Flat mention, code builds the person** (chosen). The model fills one flat schema per mention; code upserts the person and appends the day's mention. Both views, one bounded schema.

### Confirmation on the save path

1. **Everything unattended** (chosen). Consistent, no interrupts in the chain, simplest to schedule later. Mitigated by code guardrails, the eval suite, the save notice and the next-session recap.
2. **Memory keeps its yes/no.** Keeps the gate where M6 found the model least reliable, but leaves an interrupt in a chain meant to run with nobody present, so proposals would need parking as pending state.
3. **Stricter bar instead of a gate**: write only memories whose evidence is first-person. Fewer wrong writes, fewer right ones. Not chosen, but it can be added later as a guardrail without changing the design.

## Decisions made in this design

Design calls, not things the user asked for. Any can be overturned.

- **The entry extraction is the gate, not a member of the list.** The entry is the day's record; the others annotate it. Keeping it in front means "no entry, no chain", which keeps the rule simple and keeps `saved_count` and indexing where they are.
- **The chain is a service function, not graph nodes per extractor.** With no interrupts there is nothing to checkpoint between extractors, and a scheduler must be able to call it without a graph.
- **`remember_fact` keeps its confirmation.** It is a chat-time write the user asked for by name, and the interrupt costs nothing there.
- **Save-time memory extraction drops `person` items.** One home for people. `remember_fact` is unchanged.
- **The unattended memory path adds, never overwrites.** An overwrite is irreversible and the model cannot reliably tell an update from a new fact (M6). Updates stay behind the `remember_fact` confirmation.
- **Memory idempotency is best effort; people idempotency is strict.** Tagging save-path memories by day to make reruns exact would touch the memory model and repository for little gain while reruns are rare.
- **People are matched into the prompt by name, in code.** No LLM or embeddings on the chat path.
- **The exit prompt stays.** Until a scheduler exists, "Save before leaving?" is the only day-end trigger.
- **Recap is computed, not logged.** The start-of-chat recap reads the entry, memories and mentions for the most recent entry date. No run-log table.
- **The orchestrator spec stays in the repo, marked superseded,** as a record of the alternative and why it was dropped.
- **Package layout:**

```
backend/src/diary_agent/
  models/person.py                      PersonMention, PersonMentions (LLM) ; Person, Mention (stored)
  repositories/people_repository.py     people + mentions tables in journal.db
  services/extractors/
    base.py                             Extractor contract, ExtractorDeps, Report
    registry.py                         EXTRACTORS = [memory.EXTRACTOR, relationships.EXTRACTOR]
    run.py                              run_all(transcript, day, deps) -> list[Report]; never raises
    memory.py                           save-time adapter over services/memory.py
    relationships.py                    prompt, guardrails, apply
  services/agent/graph.py, nodes.py     extract -> save -> run_extractors; confirm removed
  cli/app.py                            diary people ...; start-of-chat recap; ask_to_save removed
evals/suites/relationships.py, evals/datasets/relationships.jsonl
```

## Evals

- **Baseline first.** Before any code moves, run extraction, tools, judge and simulation with `--runs 3` on the current commit and keep the results. After the change, deterministic suites (routing, retrieval) must match exactly; an LLM suite's headline metrics must land inside the baseline's min-to-max range, widened by one case. A miss means investigate before merging.
- **New `relationships` suite**, `run_evals.py --only relationships`. Transcripts with gold mentions: expected names, relation, and keywords that should appear in facts. Report **missed people** and **invented people** separately (as the tools suite reports missed and false calls), relation accuracy, facts keyword recall, and the invalid-output rate. About 20 cases following the M7 rule: hand-written for the tricky ones, AI-drafted for the bulk, reviewed by the user. Tricky cases must include: a name only the assistant said; a public figure the user watched but does not know; the user talking about themself; the same person mentioned twice; a role instead of a name ("my boss"); a group; and a transcript with no people at all.
- **Simulation suite** currently approves the save in code. It must be updated for the confirmation-free path and keep scoring the saved entry the same way.
- **Memory probe**: add a check that `person` items are dropped on the save-time path.

## Testing

Unit tests, no Ollama, fakes throughout. At minimum:

- A fake extractor registered in a test registry runs after a save and its sentence appears in the notice.
- A fake extractor that raises does not stop the next extractor, does not undo the entry, and is named in the notice.
- With no human turn, or with a failed entry extraction, no extractor runs.
- `/save` completes without an interrupt; the check-in's last step completes without an interrupt; `remember_fact` still interrupts.
- `people_repository`: applying the same day twice leaves one mention per person and no duplicate facts; `first_seen` and `last_seen` are right across days; `relation` is not downgraded to `other`.
- Relationship guardrails: a name absent from the user's turns is dropped; a name only the assistant used is dropped; an ungrounded fact is dropped.
- `people_block` includes only people named in the message and is empty otherwise.
- The save-time memory adapter drops `person` items and turns "update" proposals into "add"; the `remember_fact` path keeps both behaviours.
- Registering a new fake extractor needs no change to the graph, `run_all`, or the other extractors.

Existing tests for the removed `confirm` path are deleted; the rest keep passing.

## Left to the builder

- The exact Python shape of the contract and of `Report`.
- Exact prompt wording for relationships and the field guide text, and the name normalisation rule.
- Format and cap of the people block in the system prompt.
- The `stream_turn` status lines for the new node (today: "Looking for things to remember...").
- Which existing extraction eval transcripts can be reused as relationship cases.
- The order of migration commits, as long as every commit leaves the CLI working and tests green. Suggested: repository and models first, then the contract with memory as its only member, then remove the confirm path, then relationships, then CLI and evals.
- Updating `docs/PLAN.md` (this becomes M8; the optional polish list moves to M9), a `docs/NOTES.md` section, and the learning notes: the registry pattern in note 10, "what replaces a human gate" in notes 3 and 7, and the agent diagrams in notes 3 and the README.
