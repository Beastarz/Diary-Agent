# Save-time Extractors and Relationships Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the post-save step of the diary graph into a registry of pluggable extractors, drop the human confirmation from the save path, and add a relationships extractor that records the people the user mentions.

**Architecture:** The graph keeps `router -> extract -> save`; `confirm`, `update_memory` and the post-save `confirm_memory` hop are replaced by one node, `run_extractors`, which calls a LangGraph-free service function `run_all()`. That function walks `EXTRACTORS`, an explicit list of `Extractor` records (name, build, extract, apply, describe). Memory becomes the first member (adds only, never overwrites, drops `person` items); relationships the second, with its own `people` and `mentions` tables in `journal.db`. `confirm_memory` survives only for the chat-time `remember_fact` tool.

**Tech Stack:** Python 3.12, LangGraph 1.2.x, langchain-core, Ollama `llama3.1:8b` (structured output via `with_structured_output(method="json_schema", include_raw=True)`), SQLite, typer + rich, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-extractor-chain-design.md`

## Global Constraints

- Local only: Ollama `llama3.1:8b`, no cloud calls. Same stack as today: LangGraph 1.2.x, `SqliteSaver`, `SqliteStore`, Chroma, typer and rich.
- Layer rule: `cli/` and `api/` -> `services/` -> `repositories/` -> `models/`. `core/` usable anywhere. Nothing imports upward.
- Ollama structured output: every field required (no defaults), every list has `max_length`, field descriptions reach the model only through `field_guide()`, evidence field first.
- Code decides wherever it can; the LLM's output is bounded and checked in code.
- Prompts live in `core/prompts.py`; bump `PROMPT_VERSION` to `"v1.11"` and add a changelog line.
- One conversation thread per day, thread_id = date.
- Unit tests never call Ollama: use `ToolFakeChatModel`, `RunnableLambda` structured fakes and `KeywordEmbeddings` from `backend/tests/conftest.py`.
- Every commit leaves `uv run pytest` green and `uv run diary chat` working.
- Run `uv` and `ollama` directly (no PATH-refresh prefix).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A transcript with no people** (small talk, a day alone): the relationships extractor must report nothing and the save notice must not mention people. Test in Task 6 (`test_describe_is_empty_when_nothing_changed`, `test_guardrails_drop_everything_when_the_user_named_nobody`).
2. **Possessives and case** ("Lucas's advice", "lucas"): the name guardrail must still find the name in the user's text. Test in Task 6 (`test_guardrails_keep_a_name_the_user_wrote_in_another_form`).
3. **Ollama dies between extractors**: the entry stays saved, the next extractor still runs, and the notice names the one that failed. Test in Task 3 (`test_a_failing_extractor_does_not_stop_the_next`) and Task 5 (`test_memory_failure_does_not_undo_the_save`).
4. **Second `/save` on the same day** after more chat: one entry, one mention per person, no duplicate facts, no interrupt. Tests in Task 2 (`test_replace_mentions_twice_leaves_one_row_per_person`) and Task 5 (`test_second_save_replaces_the_entry`).
5. **A stored person whose name is inside another word** ("Ann" vs "planning"): the people block must match whole words only. Test in Task 6 (`test_people_block_matches_whole_words_only`).

---

### Task 1: Baseline eval results

No code. The anti-goal "entry extraction gets worse" needs a before picture.

**Files:**
- Read: `evals/run_evals.py`

- [ ] **Step 1: Run the default suites three times**

Run: `uv run python evals/run_evals.py --runs 3`
Expected: a scorecard per suite and `saved evals/results/evals-<timestamp>.json`.

- [ ] **Step 2: Run the opt-in suites**

Run: `uv run python evals/run_evals.py --only judge,simulation --runs 3`
Expected: same, a second results file.

- [ ] **Step 3: Record the headline numbers**

`evals/results/` is git-ignored. Copy each suite's metrics block from the two JSON files into `docs/superpowers/plans/2026-11-15-baseline.md` (a plain table: suite, metric, value) so Task 8 can compare. Commit that file:

```bash
git add docs/superpowers/plans/2026-11-15-baseline.md
git commit -m "docs(evals): record pre-extractor baseline"
```

---

### Task 2: People models and repository

**Files:**
- Create: `backend/src/diary_agent/models/person.py`
- Create: `backend/src/diary_agent/repositories/people_repository.py`
- Test: `backend/tests/repositories/test_people_repository.py`

**Interfaces:**
- Consumes: `entry_repository.connect(path) -> sqlite3.Connection` (row_factory = `sqlite3.Row`).
- Produces (`models/person.py`):
  - `Relation = Literal["family", "partner", "friend", "colleague", "acquaintance", "other"]`
  - `class PersonMention(BaseModel)`: `evidence: str`, `name: str`, `relation: Relation`, `facts: list[str]` (`max_length=3`), `what_happened: str`. All required, in this order. Each field has a `description` written for the model (see Task 6 Step 3 for the rules it must state).
  - `class PersonMentions(BaseModel)`: `mentions: list[PersonMention]` (`max_length=5`).
  - `class Person(BaseModel)`: `id: str`, `key: str`, `name: str`, `relation: Relation`, `facts: list[str]`, `first_seen: date`, `last_seen: date`, `updated_at: datetime` (default `datetime.now`).
  - `class Mention(BaseModel)`: `entry_date: date`, `person_id: str`, `what_happened: str`, `evidence: str`.
- Produces (`repositories/people_repository.py`):
  - `MAX_FACTS = 10`
  - `normalise(name: str) -> str`: lowercase, collapse whitespace, strip a leading `"my "` and trailing `"'s"`.
  - `ensure_schema(conn) -> None`: `CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY, key TEXT UNIQUE NOT NULL, name, relation, facts (JSON text), first_seen, last_seen, updated_at)` and `mentions (entry_date TEXT, person_id TEXT, what_happened TEXT, evidence TEXT, PRIMARY KEY (entry_date, person_id))`. Every public function below calls it first.
  - `upsert_person(conn, name: str, relation: Relation, facts: list[str], day: date) -> tuple[Person, bool]`: returns the stored person and `True` when it was created. Merge rules: facts appended without duplicates (compare normalised text), capped at `MAX_FACTS`; `relation` replaced only when the new value is not `"other"`; `first_seen = min`, `last_seen = max`; `name` keeps the first display form.
  - `replace_mentions(conn, day: date, mentions: list[Mention]) -> None`: delete all mentions for `day`, then insert.
  - `get_person(conn, name: str) -> Person | None` (by `normalise(name)`).
  - `list_people(conn) -> list[Person]` (by `last_seen` desc, then name).
  - `mentions_for(conn, person_id: str) -> list[Mention]` (newest first).
  - `mentions_on(conn, day: date) -> list[tuple[Person, Mention]]`.
  - `delete_person(conn, name: str) -> bool` (also deletes its mentions).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/repositories/test_people_repository.py
from datetime import date
from diary_agent.models.person import Mention
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import people_repository as repo

D1, D2 = date(2026, 11, 14), date(2026, 11, 15)

def conn(tmp_path):
    return entry_repo.connect(tmp_path / "journal.db")

def test_normalise():
    assert repo.normalise("  My Boss ") == "boss"
    assert repo.normalise("Lucas's") == "lucas"
    assert repo.normalise("Aunt  May") == "aunt may"

def test_upsert_creates_then_merges(tmp_path):
    c = conn(tmp_path)
    p, created = repo.upsert_person(c, "Lucas", "friend", ["good at math"], D1)
    assert created and p.first_seen == D1 and p.last_seen == D1
    p, created = repo.upsert_person(c, "lucas", "other", ["Good at math", "plays chess"], D2)
    assert not created
    assert p.facts == ["good at math", "plays chess"]
    assert p.relation == "friend"          # "other" never downgrades
    assert (p.first_seen, p.last_seen) == (D1, D2)
    assert p.name == "Lucas"

def test_facts_are_capped(tmp_path):
    c = conn(tmp_path)
    p, _ = repo.upsert_person(c, "Mum", "family", [f"fact {i}" for i in range(12)], D1)
    assert len(p.facts) == repo.MAX_FACTS

def test_replace_mentions_twice_leaves_one_row_per_person(tmp_path):
    c = conn(tmp_path)
    p, _ = repo.upsert_person(c, "Lucas", "friend", [], D1)
    m = Mention(entry_date=D1, person_id=p.id, what_happened="met him", evidence="met Lucas")
    repo.replace_mentions(c, D1, [m])
    repo.replace_mentions(c, D1, [m.model_copy(update={"what_happened": "met him again"})])
    [(person, mention)] = repo.mentions_on(c, D1)
    assert person.id == p.id and mention.what_happened == "met him again"
    assert [x.what_happened for x in repo.mentions_for(c, p.id)] == ["met him again"]

def test_list_get_delete(tmp_path):
    c = conn(tmp_path)
    repo.upsert_person(c, "Lucas", "friend", [], D1)
    repo.upsert_person(c, "Mum", "family", [], D2)
    assert [p.name for p in repo.list_people(c)] == ["Mum", "Lucas"]
    assert repo.get_person(c, "my mum").name == "Mum"
    assert repo.get_person(c, "nobody") is None
    assert repo.delete_person(c, "Lucas") is True
    assert repo.delete_person(c, "Lucas") is False
    assert [p.name for p in repo.list_people(c)] == ["Mum"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest backend/tests/repositories/test_people_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: diary_agent.models.person`.

- [ ] **Step 3: Implement `models/person.py` and `repositories/people_repository.py`**

Store `facts` as a JSON string like `entry_repository.LIST_FIELDS`; dates as ISO text; ids from `uuid4().hex[:8]` (same style as `memory_repository.new_id`). Module docstrings in the style of the existing repositories (one paragraph on what the tables are and why one row per person per day).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest backend/tests/repositories/test_people_repository.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/diary_agent/models/person.py backend/src/diary_agent/repositories/people_repository.py backend/tests/repositories/test_people_repository.py
git commit -m "feat(people): add people and mentions tables"
```

---

### Task 3: The extractor contract and runner

**Files:**
- Create: `backend/src/diary_agent/services/extractors/__init__.py` (empty)
- Create: `backend/src/diary_agent/services/extractors/base.py`
- Create: `backend/src/diary_agent/services/extractors/run.py`
- Create: `backend/src/diary_agent/services/extractors/registry.py`
- Test: `backend/tests/services/extractors/test_run.py` (add an empty `__init__.py` only if the other test folders have one; they do not, so skip it)

**Interfaces:**
- Produces (`base.py`):
  ```python
  @dataclass(frozen=True)
  class ExtractorDeps:
      conn: sqlite3.Connection
      store: BaseStore | None
      day: date
      model_name: str

  @dataclass
  class Report:
      name: str
      changed: int = 0
      details: list[str] = field(default_factory=list)
      error: str | None = None

  @dataclass(frozen=True)
  class Extractor:
      name: str
      build: Callable[[BaseChatModel], Runnable]
      extract: Callable[[Runnable, str], list]          # (runnable, transcript) -> items; raises on failure
      apply: Callable[[list, ExtractorDeps], Report]    # idempotent write
      describe: Callable[[Report], str]                 # one sentence, or "" when nothing changed
  ```
- Produces (`run.py`):
  - `build_runnables(model: BaseChatModel, extractors: Sequence[Extractor] = EXTRACTORS) -> dict[str, Runnable]`
  - `run_all(transcript: str, deps: ExtractorDeps, runnables: Mapping[str, Runnable], extractors: Sequence[Extractor] = EXTRACTORS) -> list[Report]`: for each extractor with a runnable in `runnables` (others are skipped silently): `items = extract(...)`, `report = apply(items, deps)`; any `Exception` from either becomes `Report(name, error=str(e))` and the loop continues. Never raises.
  - `notice_from(reports: list[Report], extractors: Sequence[Extractor] = EXTRACTORS) -> str`: for each report in order, `describe(report)` when no error, else `"(Couldn't check for {name}.)"`; non-empty parts joined with one space.
- Produces (`registry.py`): `EXTRACTORS: tuple[Extractor, ...] = ()` for now (Task 4 adds memory, Task 6 relationships). `run.py` imports `EXTRACTORS` from `registry.py`; `registry.py` imports the extractor modules; no module imports `run.py`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/services/extractors/test_run.py
from datetime import date
from langchain_core.runnables import RunnableLambda
from diary_agent.services.extractors.base import Extractor, ExtractorDeps, Report
from diary_agent.services.extractors.run import build_runnables, notice_from, run_all

DEPS = ExtractorDeps(conn=None, store=None, day=date(2026, 11, 15), model_name="fake")

def echo_extractor(name: str, log: list) -> Extractor:
    return Extractor(
        name=name,
        build=lambda model: RunnableLambda(lambda transcript: [f"{name}:{transcript}"]),
        extract=lambda runnable, transcript: runnable.invoke(transcript),
        apply=lambda items, deps: (log.extend(items), Report(name, changed=len(items), details=items))[1],
        describe=lambda r: f"{r.name} wrote {r.changed}." if r.changed else "",
    )

def broken_extractor(name: str) -> Extractor:
    def boom(*_):
        raise ConnectionError("ollama is down")
    return Extractor(name=name, build=lambda m: RunnableLambda(boom), extract=lambda r, t: r.invoke(t),
                     apply=lambda items, deps: Report(name), describe=lambda r: "never")

def test_run_all_runs_each_extractor_in_order():
    log = []
    exts = [echo_extractor("a", log), echo_extractor("b", log)]
    reports = run_all("hi", DEPS, build_runnables(None, exts), exts)
    assert log == ["a:hi", "b:hi"]
    assert [r.name for r in reports] == ["a", "b"]
    assert notice_from(reports, exts) == "a wrote 1. b wrote 1."

def test_a_failing_extractor_does_not_stop_the_next():
    log = []
    exts = [broken_extractor("a"), echo_extractor("b", log)]
    reports = run_all("hi", DEPS, build_runnables(None, exts), exts)
    assert reports[0].error == "ollama is down" and reports[0].changed == 0
    assert log == ["b:hi"]
    assert notice_from(reports, exts) == "(Couldn't check for a.) b wrote 1."

def test_an_extractor_without_a_runnable_is_skipped():
    log = []
    exts = [echo_extractor("a", log)]
    assert run_all("hi", DEPS, {}, exts) == []
    assert log == []

def test_notice_is_empty_when_nothing_changed():
    exts = [echo_extractor("a", [])]
    assert notice_from([Report("a")], exts) == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest backend/tests/services/extractors/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `base.py`, `run.py`, `registry.py`**

`base.py` module docstring explains the contract in the spec's words (five pieces, idempotent apply, never raises from `run_all`). `run.py` catches `Exception` per extractor, logs nothing (the notice is the log).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest backend/tests/services/extractors/test_run.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/diary_agent/services/extractors backend/tests/services/extractors
git commit -m "feat(extractors): add the extractor contract, registry and runner"
```

---

### Task 4: Memory as the first extractor

**Files:**
- Modify: `backend/src/diary_agent/services/memory.py` (module docstring; add `as_additions`, `remembered`)
- Create: `backend/src/diary_agent/services/extractors/memory.py`
- Modify: `backend/src/diary_agent/services/extractors/registry.py` (`EXTRACTORS = (memory.EXTRACTOR,)`)
- Test: `backend/tests/services/test_memory.py` (add), `backend/tests/services/extractors/test_memory_extractor.py` (create)

**Interfaces:**
- Consumes: `memory_service.extract_memories(extractor, transcript)`, `propose(store, items, day)`, `apply_proposals(store, proposals)`; `Extractor`, `Report`, `ExtractorDeps` from Task 3.
- Produces (`services/memory.py`):
  - `as_additions(proposals: list[dict]) -> list[dict]`: every `"update"` proposal becomes `{"action": "add", "memory": {**memory, "id": memory_repo.new_id()}, "replaces": None, "tool_call_id": ...}`; `"add"` proposals pass through.
  - `remembered(n: int) -> str`: `"Nothing remembered."` for 0, else `"Remembered N thing(s)."` (moved from `nodes._remembered`; `nodes.confirm_memory` calls this one).
- Produces (`services/extractors/memory.py`):
  - `extract(runnable, transcript) -> list[MemoryItem]`: `extract_memories(...)` then drop items with `kind == "person"`.
  - `apply(items, deps) -> Report`: `Report("memory")` with `changed=0` when `deps.store is None`; else `as_additions(propose(deps.store, items, deps.day))`, `apply_proposals`, `Report("memory", changed=len(added), details=[contents])`.
  - `describe(report) -> str`: `""` when `changed == 0`, else `remembered(report.changed)`.
  - `EXTRACTOR = Extractor(name="memory", build=build_memory_extractor, extract=extract, apply=apply, describe=describe)`.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/services/test_memory.py`:

```python
def test_as_additions_turns_updates_into_new_memories(store):
    old = remember(store, "Training to run a 10k", "goal")
    [p] = memory_service.propose(store, [item("Wants to run with family", "", "goal")], DAY)
    assert p["action"] == "update"
    [a] = memory_service.as_additions([p])
    assert a["action"] == "add" and a["replaces"] is None and a["memory"]["id"] != old.id

def test_remembered_wording():
    assert memory_service.remembered(0) == "Nothing remembered."
    assert memory_service.remembered(1) == "Remembered 1 thing."
    assert memory_service.remembered(2) == "Remembered 2 things."
```

Create `backend/tests/services/extractors/test_memory_extractor.py`:

```python
from datetime import date
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore
import pytest
from diary_agent.models.memory import Memory, MemoryCandidates, MemoryItem
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.services.extractors import memory as mem
from diary_agent.services.extractors.base import ExtractorDeps

DAY = date(2026, 11, 15)
TRANSCRIPT = "User: I'm training for a marathon. Had dinner with my sister Mia.\nAssistant: Nice!"
MARATHON = MemoryItem(evidence="I'm training for a marathon", kind="goal", content="Training for a marathon")
SISTER = MemoryItem(evidence="my sister Mia", kind="person", content="Sister named Mia")

def structured(parsed):
    return RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": parsed, "parsing_error": None})

@pytest.fixture
def store(embeddings):
    return InMemoryStore(index={"dims": 11, "embed": embeddings, "fields": ["content"]})

def deps(store):
    return ExtractorDeps(conn=None, store=store, day=DAY, model_name="fake")

def test_extract_drops_person_items():
    items = mem.extract(structured(MemoryCandidates(memories=[MARATHON, SISTER])), TRANSCRIPT)
    assert [i.content for i in items] == ["Training for a marathon"]

def test_apply_writes_without_asking_and_describes(store):
    report = mem.apply([MARATHON], deps(store))
    assert [m.content for m in memory_repo.list_memories(store)] == ["Training for a marathon"]
    assert report.changed == 1 and mem.describe(report) == "Remembered 1 thing."

def test_apply_adds_instead_of_overwriting_a_related_memory(store):
    memory_repo.put_memory(store, Memory(id="m1", kind="goal", content="Training to run a 10k", evidence="", source_date=DAY))
    related = MemoryItem(evidence="", kind="goal", content="Wants to run with family")
    mem.apply([related], deps(store))
    assert sorted(m.content for m in memory_repo.list_memories(store)) == ["Training to run a 10k", "Wants to run with family"]

def test_apply_without_a_store_changes_nothing():
    report = mem.apply([MARATHON], deps(None))
    assert report.changed == 0 and report.error is None and mem.describe(report) == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest backend/tests/services/test_memory.py backend/tests/services/extractors/test_memory_extractor.py -v`
Expected: FAIL (`as_additions`, `remembered`, module `extractors.memory` missing).

- [ ] **Step 3: Implement**

In `services/memory.py`: add `as_additions` and `remembered`; rewrite the module docstring's "Write path" paragraph: the save-time path (`extractors/memory.py`) writes without confirmation and never overwrites; `remember_fact` still confirms. In `nodes.py`, replace `_remembered` with `memory_service.remembered` (keep `confirm_memory` otherwise unchanged for now; Task 5 rewires the graph). Create `extractors/memory.py` and register it.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest`
Expected: all passed (the graph still uses `update_memory`; nothing else changed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/diary_agent/services/memory.py backend/src/diary_agent/services/extractors backend/src/diary_agent/services/agent/nodes.py backend/tests/services
git commit -m "feat(extractors): register memory as the first save-time extractor"
```

---

### Task 5: Rewire the graph and the CLI: no confirmation on the save path

**Files:**
- Modify: `backend/src/diary_agent/services/agent/state.py` (`DiaryContext`)
- Modify: `backend/src/diary_agent/services/agent/nodes.py` (remove `confirm`, `update_memory`, `route_after_update_memory`; add `run_extractors`; `route_after_extract` targets)
- Modify: `backend/src/diary_agent/services/agent/graph.py` (edges, docstring diagram, `stream_turn` status)
- Modify: `backend/src/diary_agent/cli/app.py` (`ask_to_save`/`answer_interrupt` removed, `run_turn`, `chat()` context)
- Test: `backend/tests/services/agent/test_graph.py`, `backend/tests/services/agent/test_memory_graph.py`

**Interfaces:**
- Consumes: `run_all`, `notice_from`, `build_runnables`, `ExtractorDeps` (Task 3); `extractors.memory.EXTRACTOR` (Task 4).
- Produces:
  - `DiaryContext.memory_extractor` is removed; `DiaryContext.extractors: Mapping[str, Runnable] = field(default_factory=dict)` is added (last field, frozen dataclass).
  - `nodes.run_extractors(state, runtime) -> dict`: builds `ExtractorDeps(conn=ctx.conn, store=runtime.store, day=ctx.day, model_name=ctx.model_name)`, calls `run_all(_transcript(state), deps, ctx.extractors)`, returns `{"notice": " ".join(p for p in (state.get("notice") or "Saved.", notice_from(reports)) if p)}`.
  - `nodes.route_after_extract(state) -> str`: `"save"` when `state.get("draft")` else `END`.
  - Graph: `extract -> {save | END}`, `save -> run_extractors -> END`. `confirm_memory` keeps its edge from `tools` only; its `Command` targets stay `chat_agent` / `END`.
  - `stream_turn`: the `"save" in payload` branch yields `("status", "Noting people and things to remember...")` when `context.extractors` is non-empty.
  - `REPLY_NODES` unchanged.
  - CLI: `run_turn` resumes with `Command(resume=ask_to_remember(payload))` when an interrupt is pending; after a turn whose notice starts with `"Saved."`, it prints `render_entry(entry_repo.get_entry(ctx.conn, ctx.day), ctx.day)` before the notice. The "good" colour rule becomes `notice.startswith(("Saved.", "Remembered", "Noted")) and "Couldn't" not in notice`. `chat()` builds the context with `extractors=build_runnables(get_extraction_model())`.

- [ ] **Step 1: Rewrite the affected tests in `test_graph.py`**

Replace `test_save_pauses_for_confirmation_then_writes`, `test_declining_does_not_write`, `test_second_save_reports_existing_entry`, `test_checkin_asks_each_topic_then_asks_to_save`, `test_save_through_stream_reaches_confirmation` with:

```python
def test_save_writes_without_asking(graph, ctx):
    say(graph, ctx, "Had a great dinner.")
    state = say(graph, ctx, "/save")
    assert pending_interrupt(graph, CONFIG) is None
    saved = entry_repo.get_entry(ctx.conn, DAY)
    assert saved.summary == "You ate well."
    assert saved.transcript == "User: Had a great dinner.\nAssistant: reply"
    assert state["notice"] == "Saved."
    assert state["draft"] is None
    assert state["saved_count"] == len(state["messages"])

def test_second_save_replaces_the_entry(graph, ctx):
    say(graph, ctx, "Had a great dinner.")
    say(graph, ctx, "/save")
    say(graph, ctx, "Then a walk.")
    state = say(graph, ctx, "/save")
    assert pending_interrupt(graph, CONFIG) is None
    assert entry_repo.get_entry(ctx.conn, DAY).transcript.endswith("User: Then a walk.\nAssistant: reply")
    assert len(entry_repo.list_entries(ctx.conn)) == 1
    assert state["notice"] == "Saved."

def test_checkin_asks_each_topic_then_saves(graph, ctx):
    state = say(graph, ctx, "/checkin")
    assert state["mode"] == "checkin"
    for i in range(len(CHECKIN_TOPICS) - 1):
        state = say(graph, ctx, f"answer {i}")
        assert entry_repo.get_entry(ctx.conn, DAY) is None
    state = say(graph, ctx, "last answer")
    assert state["mode"] == "chat"
    assert pending_interrupt(graph, CONFIG) is None
    assert entry_repo.get_entry(ctx.conn, DAY) is not None
    assert [m.type for m in state["messages"]].count("ai") == len(CHECKIN_TOPICS)

def test_save_through_stream_writes_the_entry(graph, ctx):
    list(stream_turn(graph, {"user_input": "Had a great dinner."}, CONFIG, ctx))
    events = list(stream_turn(graph, {"user_input": "/save"}, CONFIG, ctx))
    assert ("status", "Writing up your entry...") in events
    assert ("status", "Noting people and things to remember...") not in events  # no extractors in ctx
    assert entry_repo.get_entry(ctx.conn, DAY) is not None
```

In `test_saved_transcript_leaves_out_tool_traffic`, `test_save_indexes_the_entry_for_search` and `test_save_survives_an_indexing_failure`, delete the `graph.invoke(Command(resume=True), ...)` line and take the state from `say(graph, ctx, "/save")` where a state is needed.

- [ ] **Step 2: Rewrite `test_memory_graph.py` for the new context and path**

`make_ctx(script=None, memories=(), day=DAY, chat_model=None)` now passes `extractors={"memory": structured(MemoryCandidates(memories=list(memories)))}` instead of `memory_extractor=`. Replace `SISTER` with `MARATHON = MemoryItem(evidence="I'm training for a marathon", kind="goal", content="Training for a marathon")` and replace the three save-path tests:

```python
def test_save_writes_lasting_facts_without_asking(graph, store, make_ctx):
    ctx = make_ctx(memories=[MARATHON])
    say(graph, ctx, "I'm training for a marathon.")
    state = say(graph, ctx, "/save")
    assert pending_interrupt(graph, CONFIG) is None
    assert entry_repo.get_entry(ctx.conn, DAY) is not None
    assert [m.content for m in memory_repo.list_memories(store)] == ["Training for a marathon"]
    assert state["notice"] == "Saved. Remembered 1 thing."
    assert state["messages"][-1].type == "ai"

def test_save_drops_person_memories(graph, store, make_ctx):
    sister = MemoryItem(evidence="my sister Mia", kind="person", content="Sister named Mia")
    ctx = make_ctx(memories=[sister])
    say(graph, ctx, "Had dinner with my sister Mia.")
    state = say(graph, ctx, "/save")
    assert memory_repo.list_memories(store) == []
    assert state["notice"] == "Saved."

def test_save_drops_ungrounded_candidates(graph, store, make_ctx):
    invented = MemoryItem(evidence="I adore my cat", kind="fact", content="Has a cat")
    ctx = make_ctx(memories=[invented])
    say(graph, ctx, "Had dinner with my sister Mia.")
    state = say(graph, ctx, "/save")
    assert memory_repo.list_memories(store) == []
    assert state["notice"] == "Saved."

def test_memory_failure_does_not_undo_the_save(graph, make_ctx):
    def broken(_):
        raise ConnectionError("ollama is down")
    from dataclasses import replace
    ctx = replace(make_ctx(), extractors={"memory": RunnableLambda(broken)})
    say(graph, ctx, "Had dinner with my sister Mia.")
    state = say(graph, ctx, "/save")
    assert entry_repo.get_entry(ctx.conn, DAY) is not None
    assert state["notice"] == "Saved. (Couldn't check for memory.)"

def test_save_status_is_streamed_when_extractors_exist(graph, make_ctx):
    from diary_agent.services.agent.graph import stream_turn
    ctx = make_ctx(memories=[])
    list(stream_turn(graph, {"user_input": "Hi"}, CONFIG, ctx))
    events = list(stream_turn(graph, {"user_input": "/save"}, CONFIG, ctx))
    assert ("status", "Noting people and things to remember...") in events
```

`test_remember_fact_asks_before_writing`, `test_declined_memory_is_not_written`, `test_known_fact_skips_confirmation` and `test_memories_reach_the_prompt_on_another_day` stay as they are (the `remember_fact` interrupt must keep working).

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest backend/tests/services/agent -v`
Expected: the rewritten tests FAIL (interrupt still pending, `extractors` is an unexpected keyword).

- [ ] **Step 4: Implement the graph change**

`state.py`: swap the context field. `nodes.py`: delete `confirm`, `update_memory`, `route_after_update_memory`; add `run_extractors`; update the module docstring's save-path line to `extract (LLM) -> save (code) -> run_extractors (LLM per extractor, code writes)`. `graph.py`: nodes `extract`, `save`, `run_extractors`, `confirm_memory`; edges as in Interfaces; redraw the ASCII diagram in the docstring; change the `"save" in payload` status branch. Keep `pending_interrupt`.

- [ ] **Step 5: Implement the CLI change**

`app.py`: delete `ask_to_save` and `answer_interrupt`; `run_turn` as in Interfaces; `chat()` passes `extractors=build_runnables(get_extraction_model())` and drops `memory_service` usage if nothing else needs it. Help line stays. The exit prompt stays.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest`
Expected: all passed.

- [ ] **Step 7: Smoke the CLI by hand**

Run: `uv run diary chat`, type `I'm training for a marathon.`, then `/save`, then `exit`.
Expected: no yes/no question; the entry panel prints; a green `Saved. Remembered 1 thing.` (or `Saved.` if the model found nothing lasting); no traceback. Then `uv run diary memory list` shows the memory if one was saved.

- [ ] **Step 8: Commit**

```bash
git add backend/src/diary_agent backend/tests
git commit -m "feat(agent): run save-time extractors without confirmation"
```

---

### Task 6: The relationships extractor and the people block

**Files:**
- Modify: `backend/src/diary_agent/core/prompts.py` (`RELATIONSHIPS_SYSTEM`, `PEOPLE_SYSTEM`, `PROMPT_VERSION = "v1.11"`)
- Modify: `backend/src/diary_agent/services/extraction.py` (add `words`, `user_text`)
- Modify: `backend/src/diary_agent/services/memory.py` (use `extraction.words` / `extraction.user_text` instead of its private copies)
- Create: `backend/src/diary_agent/services/extractors/relationships.py`
- Modify: `backend/src/diary_agent/services/extractors/registry.py` (`EXTRACTORS = (memory.EXTRACTOR, relationships.EXTRACTOR)`)
- Modify: `backend/src/diary_agent/services/agent/nodes.py` (`_companion_prompt` adds the people block)
- Test: `backend/tests/services/extractors/test_relationships.py`, `backend/tests/services/agent/test_people_graph.py`

**Interfaces:**
- Consumes: Task 2 models and repository; `invoke_structured`, `field_guide` from `services/extraction.py`; `Extractor`, `Report`, `ExtractorDeps`.
- Produces (`services/extraction.py`): `words(text: str) -> list[str]` and `user_text(transcript: str) -> str`, the bodies of today's `memory._words` and `memory._user_text`.
- Produces (`core/prompts.py`):
  - `RELATIONSHIPS_SYSTEM`: tells the model it lists the people the user personally interacted with or talked about in a journaling conversation; only people the **User** named (never the Assistant); not the user themself; not public figures, celebrities or characters; one item per person; copy the user's exact words into `evidence`; `facts` only lasting things the user said about the person; empty list if no people.
  - `PEOPLE_SYSTEM`: a header line like `MEMORY_SYSTEM`'s, e.g. `"People the user has mentioned before (from their journal):\n{people}"`.
- Produces (`services/extractors/relationships.py`):
  - `MIN_FACT_OVERLAP = 0.8`, `SELF = {"i", "me", "myself", "user"}`, `MAX_BLOCK = 5`
  - `build(model) -> Runnable`: `model.with_structured_output(PersonMentions, method="json_schema", include_raw=True)`.
  - `extract_candidates(runnable, transcript, max_retries=2) -> list[PersonMention]`: `SystemMessage(f"{RELATIONSHIPS_SYSTEM}\n\nFields:\n{field_guide(PersonMention)}")` + `HumanMessage(f"Conversation:\n\n{transcript}\n\nList the people the user mentioned, if any.")` through `invoke_structured(..., "people list")`.
  - `apply_relationship_guardrails(mentions, transcript) -> list[PersonMention]`: drop when `normalise(name)` in `SELF`; drop when any word of `normalise(name)` is not in the user's words, where the user's words are `words(user_text(transcript))` with a trailing `'s` stripped from each (so "Lucas's" counts as "lucas"); keep a fact only if the share of its words longer than 2 letters that appear in the user's words is `>= MIN_FACT_OVERLAP`; merge duplicates by `normalise(name)` (facts union in order, first non-`"other"` relation, `what_happened` joined with `"; "`).
  - `extract(runnable, transcript) -> list[PersonMention]`: candidates then guardrails.
  - `apply(items, deps) -> Report`: per mention `upsert_person(deps.conn, name, relation, facts, deps.day)`, collect `Mention(entry_date=deps.day, person_id=..., what_happened=..., evidence=...)`, then `replace_mentions(deps.conn, deps.day, mentions)` (called even with an empty list, so a rerun that finds nobody clears the day). `Report("relationships", changed=len(items), details=[f"{name} (new)" if created else name])`.
  - `describe(report) -> str`: `""` for 0; `"Noted 1 person: Lucas (new)."`; `"Noted 2 people: Lucas (new), Mum."`
  - `people_block(conn: sqlite3.Connection | None, text: str) -> str`: `""` when `conn is None` (the memory probe builds a context without one, like `memory_block` with no store); otherwise people from `list_people(conn)` whose `name` appears in `text` as a whole word, case-insensitive (`re.search(rf"\b{re.escape(name)}\b", text, re.I)`), at most `MAX_BLOCK`; each line `- {name} ({relation}): {", ".join(facts) or "no details yet"}. Last seen {last_seen.isoformat()}.`; wrapped in `PEOPLE_SYSTEM`; `""` when none.
  - `EXTRACTOR = Extractor(name="relationships", ...)`.
- Produces (`nodes._companion_prompt`): the parts tuple becomes `(memory_block(...), people_block(ctx.conn, state["user_input"]), extra)`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/services/extractors/test_relationships.py
from datetime import date
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from diary_agent.models.person import PersonMention, PersonMentions
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import people_repository as people_repo
from diary_agent.services.extractors import relationships as rel
from diary_agent.services.extractors.base import ExtractorDeps

DAY = date(2026, 11, 15)
TRANSCRIPT = ("User: I met a new friend who is good at math. Lucas's advice helped.\n"
              "Assistant: How did you meet Priya?\nUser: Also saw my mum.")

def mention(name, relation="friend", facts=(), what="met", evidence="I met Lucas"):
    return PersonMention(evidence=evidence, name=name, relation=relation, facts=list(facts), what_happened=what)

def test_guardrails_keep_a_name_the_user_wrote_in_another_form():
    kept = rel.apply_relationship_guardrails([mention("lucas"), mention("Lucas's")], TRANSCRIPT)
    assert [m.name for m in kept] == ["lucas"]  # merged into one

def test_guardrails_drop_names_only_the_assistant_said_or_nobody_said():
    kept = rel.apply_relationship_guardrails([mention("Priya"), mention("Tom")], TRANSCRIPT)
    assert kept == []

def test_guardrails_drop_the_user_and_ungrounded_facts():
    kept = rel.apply_relationship_guardrails(
        [mention("me"), mention("Lucas", facts=["good at math", "plays the violin"])], TRANSCRIPT)
    assert [m.name for m in kept] == ["Lucas"]
    assert kept[0].facts == ["good at math"]

def test_guardrails_drop_everything_when_the_user_named_nobody():
    assert rel.apply_relationship_guardrails([mention("Lucas")], "User: quiet day alone.\nAssistant: Nice.") == []

def test_apply_upserts_people_and_replaces_the_days_mentions(tmp_path):
    conn = entry_repo.connect(tmp_path / "journal.db")
    deps = ExtractorDeps(conn=conn, store=None, day=DAY, model_name="fake")
    report = rel.apply([mention("Lucas", facts=["good at math"]), mention("my mum", "family")], deps)
    assert rel.describe(report) == "Noted 2 people: Lucas (new), my mum (new)."
    report = rel.apply([mention("Lucas", what="chess")], deps)
    assert rel.describe(report) == "Noted 1 person: Lucas."
    assert [(p.name, m.what_happened) for p, m in people_repo.mentions_on(conn, DAY)] == [("Lucas", "chess")]
    assert people_repo.get_person(conn, "mum") is not None  # the person stays; only the day's mention went

def test_describe_is_empty_when_nothing_changed(tmp_path):
    conn = entry_repo.connect(tmp_path / "journal.db")
    report = rel.apply([], ExtractorDeps(conn=conn, store=None, day=DAY, model_name="fake"))
    assert rel.describe(report) == "" and report.changed == 0

def test_extract_runs_the_runnable_then_the_guardrails():
    parsed = PersonMentions(mentions=[mention("Lucas"), mention("Priya")])
    runnable = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": parsed, "parsing_error": None})
    assert [m.name for m in rel.extract(runnable, TRANSCRIPT)] == ["Lucas"]

def test_people_block_matches_whole_words_only(tmp_path):
    conn = entry_repo.connect(tmp_path / "journal.db")
    people_repo.upsert_person(conn, "Ann", "friend", ["likes hiking"], DAY)
    people_repo.upsert_person(conn, "Lucas", "friend", [], DAY)
    assert rel.people_block(conn, "I'm planning a trip") == ""
    assert rel.people_block(None, "saw ann") == ""
    block = rel.people_block(conn, "saw ann and lucas")
    assert "- Ann (friend): likes hiking. Last seen 2026-11-15." in block
    assert "- Lucas (friend): no details yet. Last seen 2026-11-15." in block
```

```python
# backend/tests/services/agent/test_people_graph.py
"""Relationships through the graph: a save notes people, and a known person reaches the prompt."""
from datetime import date
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest
from diary_agent.models.entry import ExtractedEntry
from diary_agent.models.person import PersonMention, PersonMentions
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import people_repository as people_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services.agent.graph import build_graph, open_checkpointer, pending_interrupt, thread_config
from diary_agent.services.agent.state import DiaryContext

DAY = date(2026, 11, 15)
CONFIG = thread_config(DAY)
ENTRY = ExtractedEntry(highlights=[], challenges=[], gratitude=[], mood_words=[], mood=7, tags=[], summary="You met Lucas.")
LUCAS = PersonMention(evidence="met Lucas", name="Lucas", relation="friend", facts=["good at math"], what_happened="met")

def structured(parsed):
    return RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": parsed, "parsing_error": None})

@pytest.fixture
def ctx(tmp_path, embeddings, chat_model_cls):
    return DiaryContext(
        chat_model=chat_model_cls(responses=[AIMessage("reply")]), extractor=structured(ENTRY),
        conn=entry_repo.connect(tmp_path / "journal.db"), vectors=vector_repo.open_store(tmp_path / "chroma", embeddings),
        day=DAY, model_name="fake", extractors={"relationships": structured(PersonMentions(mentions=[LUCAS]))})

@pytest.fixture
def graph(tmp_path):
    return build_graph(open_checkpointer(tmp_path / "checkpoints.db"))

def test_save_notes_people(graph, ctx):
    graph.invoke({"user_input": "I met Lucas, who is good at math."}, CONFIG, context=ctx)
    state = graph.invoke({"user_input": "/save"}, CONFIG, context=ctx)
    assert pending_interrupt(graph, CONFIG) is None
    assert state["notice"] == "Saved. Noted 1 person: Lucas (new)."
    assert people_repo.get_person(ctx.conn, "Lucas").facts == ["good at math"]

def test_known_person_reaches_the_prompt(graph, ctx, chat_model_cls):
    class Recording(chat_model_cls):
        seen: list = []
        def _generate(self, messages, *args, **kwargs):
            self.seen.append(messages)
            return super()._generate(messages, *args, **kwargs)
    people_repo.upsert_person(ctx.conn, "Lucas", "friend", ["good at math"], date(2026, 11, 10))
    from dataclasses import replace
    model = Recording(responses=[AIMessage("How is Lucas?")])
    graph.invoke({"user_input": "Lucas called today."}, CONFIG, context=replace(ctx, chat_model=model))
    assert "- Lucas (friend): good at math. Last seen 2026-11-10." in model.seen[-1][0].content
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest backend/tests/services/extractors/test_relationships.py backend/tests/services/agent/test_people_graph.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Prompts first (`RELATIONSHIPS_SYSTEM`, `PEOPLE_SYSTEM`, version bump with a `# v1.11:` line). Then `extraction.words` / `extraction.user_text` and point `memory.py` at them (delete its private copies). Then `relationships.py` as in Interfaces, with a module docstring that states the guardrail rules and why (the name rule replaces the human gate). Register it. Add the people block to `_companion_prompt`. The `PersonMention` field descriptions (Task 2) must now say: `evidence` = the user's exact words; `name` = how the user refers to them; `relation` = one of the six values, `other` if unclear; `facts` = up to 3 lasting things the user said about them, empty if none; `what_happened` = one sentence about today's interaction.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest`
Expected: all passed.

- [ ] **Step 5: Smoke the CLI by hand**

Run: `uv run diary chat`, type `I met Lucas today, a new friend who is good at math.`, `/save`, `exit`.
Expected: `Saved. ... Noted 1 person: Lucas (new).` Then a new `uv run diary chat`, type `Lucas texted me.`: the reply should show the model knows Lucas (the block is in the prompt; the wording is the model's).

- [ ] **Step 6: Commit**

```bash
git add backend/src/diary_agent backend/tests
git commit -m "feat(relationships): extract the people the user mentions at save time"
```

---

### Task 7: CLI: `diary people` and the start-of-chat recap

**Files:**
- Create: `backend/src/diary_agent/services/recap.py`
- Modify: `backend/src/diary_agent/cli/app.py`
- Test: `backend/tests/services/test_recap.py`

**Interfaces:**
- Consumes: `entry_repo.list_entries(conn, limit=1)`, `memory_repo.list_memories(store)`, `people_repo.mentions_on(conn, day)`.
- Produces (`services/recap.py`): `recap(conn, store: BaseStore | None) -> str | None`. `None` when there is no entry. Otherwise, for the newest entry's date `d`: `n` = memories whose `source_date == d` (0 when `store is None`), `people` = names from `mentions_on(conn, d)`. Text: `f"Last entry: {d:%a %d %b}."` + `f" Remembered {n} thing(s) (diary memory list)."` when `n` + `f" Noted {names joined with ', '} (diary people list)."` when people. Use `remembered()`'s singular/plural rule.
- CLI:
  - `chat()` prints the recap in dim after the model line, before the resume line.
  - `people_app = typer.Typer(help="People you've mentioned in your journal.", no_args_is_help=True)` registered as `people`:
    - `list`: table `name, relation, facts, first seen, last seen`; dim hint when empty.
    - `show <name>`: the person line plus one row per mention (`date, what happened`); exit 1 when unknown.
    - `forget <name>`: `delete_person`; exit 1 when unknown; prints `Forgotten.`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/services/test_recap.py
from datetime import date
import pytest
from langgraph.store.memory import InMemoryStore
from diary_agent.models.entry import ExtractedEntry, JournalEntry
from diary_agent.models.memory import Memory
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.repositories import people_repository as people_repo
from diary_agent.models.person import Mention
from diary_agent.services.recap import recap

D = date(2026, 11, 14)
ENTRY = ExtractedEntry(highlights=[], challenges=[], gratitude=[], mood_words=[], mood=6, tags=[], summary="ok")

@pytest.fixture
def conn(tmp_path):
    return entry_repo.connect(tmp_path / "journal.db")

def save(conn, day):
    entry_repo.save_entry(conn, JournalEntry(**ENTRY.model_dump(), entry_date=day, transcript="", model="f", prompt_version="t"))

def test_no_entry_means_no_recap(conn):
    assert recap(conn, None) is None

def test_recap_names_the_last_day_and_what_was_written(conn, embeddings):
    store = InMemoryStore(index={"dims": 11, "embed": embeddings, "fields": ["content"]})
    save(conn, date(2026, 11, 10))
    save(conn, D)
    memory_repo.put_memory(store, Memory(id="m1", kind="goal", content="Marathon", evidence="", source_date=D))
    memory_repo.put_memory(store, Memory(id="m2", kind="goal", content="Older", evidence="", source_date=date(2026, 11, 10)))
    p, _ = people_repo.upsert_person(conn, "Lucas", "friend", [], D)
    people_repo.replace_mentions(conn, D, [Mention(entry_date=D, person_id=p.id, what_happened="met", evidence="")])
    assert recap(conn, store) == "Last entry: Sat 14 Nov. Remembered 1 thing (diary memory list). Noted Lucas (diary people list)."

def test_recap_without_store_or_people(conn):
    save(conn, D)
    assert recap(conn, None) == "Last entry: Sat 14 Nov."
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest backend/tests/services/test_recap.py -v`
Expected: FAIL with `ModuleNotFoundError: diary_agent.services.recap`.

- [ ] **Step 3: Implement `services/recap.py` and the CLI additions**

- [ ] **Step 4: Run the suite and the CLI**

Run: `uv run pytest`
Expected: all passed.
Run: `uv run diary people list`, `uv run diary people show Lucas`, `uv run diary chat` (then `exit`).
Expected: the table from Task 6's smoke test; the person with one mention; the recap line at the start of chat.

- [ ] **Step 5: Commit**

```bash
git add backend/src/diary_agent/services/recap.py backend/src/diary_agent/cli/app.py backend/tests/services/test_recap.py
git commit -m "feat(cli): add diary people and a start-of-chat recap"
```

---

### Task 8: Evals: relationships suite, simulation fix, baseline comparison

**Files:**
- Modify: `evals/evaluators.py` (add `check_relationships`)
- Create: `evals/suites/relationships.py`, `evals/datasets/relationships.jsonl`
- Modify: `evals/run_evals.py` (`SUITES` gains `"relationships"`; docstring line)
- Modify: `evals/suites/simulation.py` (`converse`)
- Modify: `evals/memory_probe.py` (save-path check)
- Test: `evals/tests/test_evaluators.py` (add), `evals/tests/test_datasets.py` (add)

**Interfaces:**
- Dataset row: `{"id": str, "transcript": str, "people": [{"name": str, "relation": Relation, "facts": [pattern, ...]}], "allowed": [name, ...]}`. `facts` and `allowed` optional. `people` may be `[]`. Patterns follow `matches()` ("a|b" = either, substring).
- `check_relationships(case: dict, mentions: list) -> dict[str, bool]`: keys `found` (every expected name present, compared with `people_repository.normalise`), `no_invented` (no mention outside expected + allowed), `relation` (every found expected person has the expected relation; only when the case lists people), `facts` (every expected fact pattern matches that person's facts; only for people that list facts).
- Suite metrics: `valid`, `all_checks`, `found`, `no_invented`, `relation`, `facts` (rates), `n_missed`, `n_invented` (counts over all runs). `texts` = every `what_happened` and fact, for the language suite. Pattern after `suites/extraction.py`; the runnable comes from `relationships.build(get_extraction_model())`, the call is `relationships.extract(runnable, transcript)`.

- [ ] **Step 1: Write the failing evaluator and dataset tests**

Add to `evals/tests/test_evaluators.py`:

```python
def test_check_relationships():
    from evaluators import check_relationships
    from diary_agent.models.person import PersonMention
    case = {"id": "x", "transcript": "", "people": [{"name": "Lucas", "relation": "friend", "facts": ["math"]}], "allowed": ["Mum"]}
    lucas = PersonMention(evidence="", name="lucas", relation="friend", facts=["good at math"], what_happened="")
    mum = PersonMention(evidence="", name="my mum", relation="family", facts=[], what_happened="")
    tom = PersonMention(evidence="", name="Tom", relation="other", facts=[], what_happened="")
    assert check_relationships(case, [lucas, mum]) == {"found": True, "no_invented": True, "relation": True, "facts": True}
    assert check_relationships(case, [tom]) == {"found": False, "no_invented": False, "relation": True, "facts": True}
    assert check_relationships({"id": "y", "transcript": "", "people": []}, []) == {"found": True, "no_invented": True}
```

Add to `evals/tests/test_datasets.py`:

```python
RELATIONS = {"family", "partner", "friend", "colleague", "acquaintance", "other"}

def test_relationship_cases_are_well_formed():
    rows = load_jsonl("relationships.jsonl")
    assert unique_ids(rows) and len(rows) >= 20
    for r in rows:
        assert set(r) <= {"id", "transcript", "people", "allowed"}, r["id"]
        assert r["transcript"].startswith("User:"), r["id"]
        for p in r["people"]:
            assert p["relation"] in RELATIONS, r["id"]
            assert p["name"].lower() in r["transcript"].lower(), f"{r['id']}: {p['name']} is not in the user's text"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest evals/tests -v`
Expected: FAIL (`check_relationships` missing, dataset missing).

- [ ] **Step 3: Implement `check_relationships` and write the dataset**

The dataset has at least 20 cases. These seven are required, hand-written, each with its own id: a name only the assistant said (`people: []`, transcript where the Assistant asks about "Priya"); a public figure the user watched (`people: []`); the user talking about themself only (`people: []`); the same person mentioned twice across two user turns (one expected person); a role instead of a name (`"my boss"`, relation `colleague`); a group (`"my team"` or `"the kids"`); a transcript with no people at all (small talk). Draft the remaining cases to cover each relation value at least twice and at least five cases with `facts`. Mark the file's header comment with the M7 rule: hand-written tricky cases first, AI-drafted bulk after, reviewed by the user. **Stop and ask the user to review the dataset before Step 5.**

- [ ] **Step 4: Write the suite, register it, fix the simulation suite and the probe**

`suites/relationships.py` with a module docstring like the extraction suite's. `run_evals.py`: `SUITES = ("routing", "retrieval", "extraction", "relationships", "tools")` and a docstring line. `suites/simulation.py::converse`: the loop stops only when `state["messages"][-1].type != "ai"`; delete the `pending_interrupt` import and the resume block; the entry is read from `journal.db` as before (`real_graph` passes no `extractors`, so `run_extractors` writes nothing). `memory_probe.py::run_extract`: alongside raw and kept, print `save-path` = `extractors.memory.extract(...)` so the dropped `person` items are visible.

- [ ] **Step 5: Run the tests, then the evals**

Run: `uv run pytest`
Expected: all passed.
Run: `uv run python evals/run_evals.py --runs 3` then `uv run python evals/run_evals.py --only judge,simulation --runs 3`.
Expected: the `relationships` scorecard prints; routing and retrieval metrics equal Task 1's exactly; every other LLM suite's headline metrics fall inside Task 1's min-to-max range widened by one case. If not, investigate before committing and record what was found in the commit message.

- [ ] **Step 6: Append the numbers to the baseline file and commit**

Add an "after" column to `docs/superpowers/plans/2026-11-15-baseline.md` with the relationships suite's first numbers.

```bash
git add evals docs/superpowers/plans/2026-11-15-baseline.md
git commit -m "feat(evals): add the relationships suite and drop the save confirmation from simulation"
```

---

### Task 9: Docs and learning notes

**Files:**
- Modify: `docs/PLAN.md`, `docs/NOTES.md`, `README.md`
- Modify: `docs/learning/README.md`, `docs/learning/03-langgraph.md`, `docs/learning/07-memory.md`, `docs/learning/10-agent-design-principles.md`

- [ ] **Step 1: PLAN.md**

Status table: `M8 Save-time extractors and relationships | ✅ done`, polish list renumbered to M9. Change log entry dated 2026-11-15 in the style of the others: one agent kept, orchestrator design superseded, the extractor registry, no confirmation on the save path (why: future scheduled run), memory adds-only, relationships tables, `diary people`, the relationships eval suite and its first numbers. Project layout: add `models/person.py`, `repositories/people_repository.py`, `services/extractors/`, `services/recap.py`, `evals/suites/relationships.py`, the dataset. Design principles: "Human confirmation before writing an entry or a memory" becomes "Chat-time writes (`remember_fact`) are confirmed; save-time writes are unattended, guarded by code and measured by evals". Milestone section **M8** written like M3 to M7, with a *Learn:* line.

- [ ] **Step 2: NOTES.md**

New section `## M8: save-time extractors and relationships` with the same sub-headings as M6: what was built, what the eval found (Task 8 numbers), Lessons (at least: registry instead of orchestrator and why; what replaces a human gate; idempotent writes; name-in-user-text as the strongest guardrail), Known limits (memory idempotency is best effort; no migration of old `person` memories; re-extraction only on `/save`), Try this.

- [ ] **Step 3: README and learning notes**

README roadmap: tick M7, add `- [x] M8 Save-time extractors + relationships (`diary people`)`. `docs/learning/README.md`: fix the "agent at a glance" diagram (`extract → save → run_extractors [memory, relationships]`; `remember_fact → confirm_memory`). `03-langgraph.md`: point 1 node list, point 4 and point 7 now cite `confirm_memory` only, and a sentence on why `confirm` was removed. `07-memory.md`: point 4 becomes "Two write paths, one gate left" (save-time path unattended, adds only; `remember_fact` confirmed), point 5 gains the recap line as the mitigation. `10-agent-design-principles.md`: point 4 retitled "Confirm what the user asked for by name; measure what runs unattended", and a new point 13 "A registry beats an orchestrator when domains diverge at the edges" linking to the superseded spec. Keep the three-part layout (Concept, In this repo, Why it matters) and the `Try it` lists.

- [ ] **Step 4: Commit**

```bash
git add docs README.md
git commit -m "docs: record M8 save-time extractors and relationships"
```
