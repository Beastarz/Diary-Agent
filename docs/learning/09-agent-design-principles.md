# 9. Agent design principles: the patterns that kept coming back

The other notes are organised by technology. This one is organised by **lesson**. Each principle below showed up in several milestones, usually as a bug first and a fix second. If you only read one note, read this one.

---

### 1. When a prompt rule doesn't hold, move the decision into code
The single most repeated lesson. The model is good at language and judgement; code is better at anything checkable.

| Rule we wanted | Prompt-only result | Code fix | Result |
|---|---|---|---|
| Only list gratitude the user expressed (M2) | prompt tweak broke another case | `extraction.apply_guardrails` | 100%, can't regress |
| No tool calls on small talk (M4) | 75% → 75% false calls | `LOOKUP_REQUEST` gate in `tools_for` | 0% |
| Get "Monday this week" right (M4) | wrong dates | `calendar_hint` cheat-sheet | date args 50% → 90% |
| Don't claim a trend from 2 days (M5) | 0/3 | `summary.apply_guardrails` writes `mood_trend` | fixed |
| Don't save facts about your sister as yours (M6) | — | third-person rule in `apply_memory_guardrails` | pollution 1 → 0 |
| Don't save "I remember the beach trip" (M6) | 4/4 false saves | `MEMORY_REQUEST` per-tool gate | 0/4 |
| Keep past days out of today's entry (M7) | fixed 1 case, broke 2 | `format_transcript` drops post-lookup replies | kept |

**Takeaway:** Let the model pick from a menu the code built. Let the model interpret numbers the code computed.

### 2. Never ask the model for what code already knows
Dates, the model name, the prompt version, averages, counts, which messages came from a tool lookup. Each is free and exact in code, and an extra way to fail if you ask the model. Two-layer schemas (`ExtractedEntry` → `JournalEntry`, `PeriodSummary` → `SummaryRecord`, `MemoryItem` → `Memory`) make this the default. See [note 2](02-structured-output.md), point 7.

### 3. Guard the context: what goes in comes back out
Extraction reads the conversation, so anything that enters `messages` can end up in today's diary entry. Each of these was a real or near-miss contamination path, and each has its own fix:

| Would have leaked | Fix |
|---|---|
| `/save`, `/cancel` commands | handled by the router, never in `messages`; status in `notice` |
| Tool results (past entries) | `format_transcript` drops tool messages |
| The assistant retelling a past day after a lookup | `format_transcript` drops the post-lookup reply |
| A weekly review | summaries run in the CLI, not in chat |
| Long-term memories | injected into the system prompt, not `messages` |

Still open: the assistant recalling a past day *without* a lookup (e.g. from a memory) is still extracted as today's event ([issue #9](https://github.com/Beastarz/Diary-Agent/issues/9)).

### 4. A human approves every write
Entries (`confirm` → `save`), summaries (Confirm before `save_summary`), memories (`confirm_memory`). Tools are read-only or only *propose*. The model never writes without a yes. Mechanically, that means `interrupt()` plus the rule "no side effects before `interrupt()`" ([note 3](03-langgraph.md), point 7).

### 5. Bound everything the model controls
Lists (`max_length`), output length (`num_predict`), tool rounds (`MAX_TOOL_ROUNDS`), collapse rounds (`MAX_COLLAPSE_ROUNDS`), memory length (120 chars), injected memories (15). Every unbounded thing eventually ran away: tags looped until the probe hung for 10 minutes.

### 6. One source of truth; everything else fails soft
SQLite holds the truth. The vector index and embeddings are derived and rebuildable. If indexing, embedding or memory extraction fails, the entry is still saved and the user gets a hint ("Run 'diary reindex' later"). Secondary features must never block the primary job.

### 7. Every LLM hop is a chance to distort, so use fewer hops
The pottery distortion came from an extra LLM map step that re-condensed what extraction had already condensed (and the user had approved). Replacing it with code was faster (15 s → 4.3 s) *and* better. Before adding a step, ask whether an earlier step already produced what you need ([note 6](06-context-and-summaries.md), point 4).

### 8. Prompt changes are trades. Measure the net
Prompt fixes moved failures around more often than they removed them (v1.4 in M2, "today only" in M7). Version every prompt (`PROMPT_VERSION`), rerun the *whole* suite, and compare variants side by side with repeats ([note 8](08-evals.md)).

### 9. Small models have biases; know yours
llama3.1 over-calls tools; qwen3 under-called them and leaked CJK characters. 8B models can't do date maths, pad text to hit length targets ("2-3 sentences" → invention), and narrate whatever facts you put in front of them. Design around the biases you've *measured*, not the ones you assume.

### 10. Local-first means privacy by construction
Everything lives in `data/`, which is git-ignored, and a CI `privacy` job fails if journal data, eval results or secrets (`data/`, `*.db`, `chroma/`, `.env`) are ever tracked, even after a `git add -f`. A diary must never be published by accident.

---

### Open problems worth trying yourself
- The keyword router starts a check-in on "I had to check in at the hotel" (M7 routing: route 82%, tool gate 62%).
- `MAX_DISTANCE` has no margin at 60 entries (0.41 vs 0.41). Is there a better "nothing found" signal than one threshold?
- Value-change memory updates ("moved to Leeds") get added as a second memory. Would an LLM "same fact?" check work, and how would you eval it?
- The summary overview still overgeneralises from sparse data ("great month" from 1 entry). The planned M7 LLM judge is meant to measure this.
- Issue #9: past-day recall without a tool lookup leaks into today's entry.

**Full story:** [PLAN.md](../PLAN.md) → "Design principles", and the "Lessons" sections of each milestone in [NOTES.md](../NOTES.md).
