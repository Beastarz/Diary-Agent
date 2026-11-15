# Orchestrator and modular agents — design

Date: 2026-10-27. Status: **superseded** on 2026-11-15 by `2026-11-15-extractor-chain-design.md`. Kept as a record of the multi-agent alternative and why it was dropped: in this app the domains share one conversation and only diverge at save time.

## Intent

Today the diary is one LangGraph graph: a rule-based router sends each turn to chat, check-in, summary or save. Everything shares one state, one entries table and one memory namespace.

The goal is one chatbot that works like Siri opening apps. The user talks to a single chat and never picks a log. A central **orchestrator** works out which **agent** (an "app") each part of a message belongs to and hands it over. The diary becomes the first agent. Later agents are a work log, expense tracking, a relationship log and calendar planning. Each new one should be a new folder plus one line of registration, not a rewrite.

A single message can hold several jobs. "Tense meeting with my boss, then dinner with Sam, spent $40" should become three jobs, queued and handled in turn.

This is also a learning build. It should teach orchestrator / multi-agent design, subgraphs, and how to eval an LLM planner on an 8B local model.

## How the orchestrator and agents interact

This section is the heart of the design. Everything else follows from it.

### The roles

- **Orchestrator.** Owns the conversation the user sees, the job queue, and the turn loop. It never does an agent's work. It decides *who* handles a piece of text, never *what* to do with it.
- **Planner.** The part of the orchestrator that turns one user message into an ordered list of jobs.
- **Agent.** A self-contained LangGraph subgraph with its own state, prompts, tools, storage and interrupts. It receives a job, does the work, and reports back with a small fixed result. Agents never call each other.
- **Registry.** One explicit list of the installed agents. The planner only knows the agents listed there.

### A turn, step by step

```
            user message
                 |
                 v
   +----------------------------+
   | 1. orchestrator command?   |  code: /cancel, /help
   +----------------------------+
                 | no
                 v
   +----------------------------+     yes
   | 2. does an agent hold the  |-------------> send the whole message to that agent
   |    floor? (mid-flow job)   |               (no planning), go to 4
   +----------------------------+
                 | no
                 v
   +----------------------------+
   | 3. planner                 |  a. slash command owned by an agent -> that agent (code)
   |                            |  b. only one agent registered -> that agent (code)
   |                            |  c. otherwise LLM: message -> [Job, Job, ...]
   |                            |  d. any failure -> one job for the default agent
   +----------------------------+
                 |
                 v
   +----------------------------+
   | 4. dispatcher: run the     |<------------------------------+
   |    job at the queue head   |                               |
   +----------------------------+                               |
                 |                                              |
                 v                                              |
        agent subgraph runs                                     |
        (may stream a reply, may interrupt() for a yes/no)      |
                 |                                              |
                 v                                              |
          AgentResult.status                                    |
       +---------+------------------+                           |
       | done / failed              | needs_input               |
       v                            v                           |
  pop the job.               the job stays at the head and      |
  more jobs queued? -- yes --------------------------------- ---+
       | no                  holds the floor. The turn ends.
       v                     The next user message goes
      END                    straight to it (step 2).
```

### The evening example

The user types: "Tense meeting with my boss, then dinner with Sam, spent $40." Assume the work log and expenses agents exist by then.

1. No agent holds the floor, so the planner runs. It returns three jobs: `work_log: "Tense meeting with my boss"`, `diary: "dinner with Sam"`, `expenses: "spent $40"`.
2. The orchestrator posts one line written in code, not by the LLM: "Got 3 things: work log, diary, expenses. Starting with the work log."
3. The work log agent replies with a follow-up question and returns `needs_input`. It now holds the floor.
4. The user answers. The message goes straight to the work log. It saves, after its own yes/no interrupt, and returns `done`.
5. The dispatcher moves on to the diary job in the same turn, and so on until the queue is empty.

With only the diary installed, step 3b sends every message to the diary. The bot behaves exactly as it does today.

### The contract

Every agent is described by an `AgentSpec` and talks to the orchestrator only through `AgentInput` and `AgentResult`. These three types are the whole interface. A change to them is a change to every agent, so keep them small.

**AgentSpec** (what an agent tells the registry):

| Field | Purpose |
|---|---|
| `name` | Stable id, e.g. `diary`. Used in the job queue, checkpoints and interrupt payloads. |
| `description` | One or two sentences saying *when* to use this agent. The planner reads it. Written like a tool docstring. |
| `examples` | A few short example messages. They go in the planner prompt and seed the planner eval. |
| `commands` | Slash commands this agent owns, e.g. `/checkin`, `/save`, `/summary`. The planner routes these in code. |
| `default` | True for exactly one agent, the fallback for chat and planner failures. The diary is the default. |
| `reply_nodes` | Which of its nodes stream tokens to the user (today's `REPLY_NODES`). |
| `build` | Returns the compiled subgraph, given the shared store. |
| `make_context` | Builds the agent's own runtime context from the shared context. |

| `unsaved` | Optional. Given the agent's own state, returns a question such as "Save today's entry before leaving?" when it has unsaved work, else nothing. The CLI asks it on exit. |
| `save_command` | Optional. The command the CLI runs when the user says yes to `unsaved`, e.g. `/save`. |

**AgentInput** (what the orchestrator sends):

| Field | Meaning |
|---|---|
| `task` | The user's own words for this job, copied verbatim, never paraphrased. For a floor-holding agent, it is the user's whole new message. |
| `job_id` | Identifies the job in the queue. |
| `cancel` | True when the user typed `/cancel` while this agent held the floor. The agent resets its own flow (the diary leaves check-in mode and says "Check-in cancelled.") and returns `done`. The orchestrator never touches an agent's state from outside. |

**AgentResult** (what the agent returns):

| Field | Meaning |
|---|---|
| `status` | `done`: finished, release the floor. `needs_input`: waiting on the user, keep the floor. `failed`: give up on this job, tell the user, move on. |
| `reply` | The text the user saw, if any. The orchestrator copies it into its own transcript. |
| `notice` | A one-line status for the CLI, like today's "Saved." |

### What is shared and what is owned

- **Owned by each agent:** its state schema, its own message history, prompts, tools, storage (the diary keeps `journal.db` and Chroma), and its interrupts.
- **Shared, passed in by the orchestrator:** the chat model, today's date, the model name, and the long-term memory store.
- **Long-term memory:** the `("user", "profile")` namespace stays shared, so every agent can read who the user is. Writes still go through a human yes/no, as in M6. An agent that needs private long-term data uses its own namespace, `("agent", <name>, ...)`.
- **Conversation history:** the orchestrator keeps what the user saw. Each agent keeps only the turns that were routed to it. This matters for the diary: extraction then reads only diary-related text, so "spent $40" never lands in a journal entry.

### Interrupts

Agents keep calling `interrupt()` as today. The interrupt rises through the orchestrator to the CLI. Its payload must carry the agent's `name`, so the CLI can pick that agent's renderer. Today's `ask_to_save` and `ask_to_remember` become the diary's renderers. The resume value goes back to the same agent unchanged.

## Goals

- One orchestrator graph that the CLI talks to instead of the diary graph.
- The diary moved behind the agent contract with no change in behaviour.
- Adding an agent means a new package folder plus one line in the registry, and its CLI renderer if it interrupts. No edits to the orchestrator.
- A planner that splits one message into several jobs, measured by a new eval suite before a second real agent exists.
- Tests that prove the contract with fake agents: queueing, floor holding, interrupt pass-through, and state that survives across turns.

## Non-goals

- Building the work log, expenses, relationship or calendar agents. This design only leaves room for them.
- Running jobs in parallel. Jobs run one at a time, in order.
- Agents calling each other, or questions that span agents ("what did I spend at dinner with Sam?").
- The HTTP API and web frontend.
- Switching agents in the middle of a flow. If the user mentions an expense mid check-in, it goes to the diary. `/cancel` is the way out for now.

## Anti-goals

These would make the work a failure even if it runs:

- **The diary gets worse.** Every existing eval suite must score within normal run-to-run noise of its last results.
- **Plain chat gets slower** while only the diary is installed. That is why the planner skips the LLM when there is one agent.
- **A wrong agent writes something.** Writes always stay behind each agent's own yes/no.
- **Adding an agent needs orchestrator edits.** If the second agent can't be added by registration alone, the contract is wrong.

## Constraints

- Local only: Ollama, `llama3.1:8b`, no cloud calls. Same stack as today: LangGraph 1.2.x, `SqliteSaver`, `SqliteStore`, Chroma, typer and rich.
- The layer rule holds: `cli/` and `api/` call `services/`, which call `repositories/`, which use `models/`.
- The 8B model is weak at open-ended decisions; the routing evals already showed it. So code decides wherever it can, and the planner's LLM output is bounded and checked in code. From M2: Ollama structured output needs all fields required and lists with a maximum length.
- One conversation thread per day, as today (thread_id = date).
- Prompts stay versioned in `core/prompts.py`. The planner prompt gets its own entry there.

## Decisions made in this design

These are design calls, not things the user asked for. Any of them can be overturned.

- **Orchestrator, not a bigger router.** A router sends a message to one place. Splitting one message into several jobs needs a planner and a queue.
- **Jobs run one at a time, and each agent asks its own questions and its own yes/no.** The alternative was one combined approval at the end. One at a time reuses today's confirm flow and keeps each agent self-contained.
- **Floor holding.** An agent waiting on the user gets the next message directly, like an open app on Siri. This is how today's multi-turn check-in keeps working.
- **Planner order: code first, LLM last.** Orchestrator commands, then the floor, then agent slash commands, then the single-agent shortcut, then the LLM.
- **Tasks are verbatim.** The planner copies spans of the user's words; it never rewrites them. The diary's extraction and saved transcript must read what the user actually wrote.
- **One job per agent per message.** If two separate parts of a message belong to the same agent, the planner joins their spans into one job. Jobs run in the order each agent is first mentioned.
- **Leftover text goes to the default agent.** Words no job claims, such as "Ugh, long day" or "thanks!", are added to the default agent's job, creating one if needed. Leftovers that are only punctuation or filler are dropped.
- **Planner guardrails in code.** An unknown agent name maps to the default agent. A task that isn't a near-verbatim span of the message is replaced by the matching text or dropped. An empty or failed plan becomes one default job with the whole message. At most 4 jobs per message.
- **`/cancel` is forwarded.** The orchestrator sends `cancel=True` to the floor holder, then clears the floor and the rest of the queue. With no floor holder it just clears the queue.
- **Exit and resume stay generic.** "Resuming today's conversation" and re-showing a pending question come from the orchestrator's own transcript. "Save before leaving?" comes from each agent's optional `unsaved` hook, so a future agent with drafts needs no CLI edits.
- **Handoff lines are written in code,** e.g. "Got 3 things: ...". No LLM call for them.
- **Explicit registry,** a plain list in one file. No plugin auto-discovery.
- **The diary is the default agent.** It handles small talk and anything the planner can't place.
- **Fresh checkpoint file.** The new graph's state doesn't match old checkpoints. Today's unsaved conversation at the moment of migration is lost. Saved entries and memories are untouched.
- **Package layout:**

```
backend/src/diary_agent/services/
  orchestrator/
    contract.py     AgentSpec, AgentInput, AgentResult
    registry.py     AGENTS = [diary.SPEC]
    planner.py      code rules + LLM plan + guardrails
    state.py        conversation, job queue, floor holder
    graph.py        planner -> dispatcher -> agent nodes loop
  agents/
    diary/          today's services/agent/, moved behind the contract
      spec.py       SPEC = AgentSpec(...)
      graph.py  nodes.py  state.py  tools.py
backend/src/diary_agent/cli/
  app.py            talks to the orchestrator
  renderers/diary.py  ask_to_save, ask_to_remember
```

## Evals

- **Regression:** routing, tools, extraction, language, judge and simulation keep running. Routing now tests the diary's internal router. Tools and simulation go through the orchestrator.
- **Baseline first.** Before the diary moves, run each LLM suite with `--runs 3` on the current commit and keep those results. After the move, a deterministic suite (routing, retrieval) must match exactly. An LLM suite's headline metrics must land inside the baseline's min-to-max range, widened by one case. A miss means investigate before merging, not an automatic revert.
- **New planner suite,** `run_evals.py --only planner`. It runs the LLM planner against a registry of *description-only* stub agents (work log, expenses, relationships, calendar), so splitting can be measured before those agents exist. Each case is a message and its expected ordered jobs, each with its agent and verbatim task. Report: exact match on the ordered agent list, set match ignoring order, missed and extra agents separately, task overlap with the gold span, and fallback rate. The dataset follows the M7 rule: hand-written for the tricky cases, AI-drafted for the bulk, reviewed by the user. It must include single-intent messages, small talk, and messages that only look multi-part.

## Testing

Unit tests with fake agents, no Ollama. At minimum: an echo agent that returns `done`; an agent that needs two turns, returning `needs_input` and then `done`; and an agent that calls `interrupt()`. They must prove:

- several jobs from one message run in order within one turn;
- floor holding sends the next message to the waiting agent and skips the planner;
- an interrupt reaches the caller with the agent's name, and the resume value reaches the same agent;
- an agent's own state survives across turns and across a restart with the same checkpointer;
- `/cancel` reaches the floor holder as `cancel=True`, and the floor and queue are cleared;
- an agent's `unsaved` hook is asked on exit and its `save_command` runs on yes;
- registering a new fake agent needs no orchestrator change.

The existing diary tests move with the diary and keep passing.

## Left to the builder

- The LangGraph mechanics of embedding agents. Options include adding each compiled subgraph as a node, or calling it from a wrapper node with its own `context=`. Whatever is chosen must keep interrupts, per-agent state across turns, token streaming, and a separate runtime context per agent. **Prove this with a small spike test before moving the diary.** Per-agent persistence may need subgraphs compiled with `checkpointer=True`. Check that an agent can run a new job in the same turn it finished one, e.g. after `/cancel` or when a held-floor job ends and the queue continues.
- How streaming tags tokens with their agent, given the subgraph nesting.
- Exact planner prompt wording and the planner model's settings.
- Whether `/summary`, which only prints a pointer to the CLI today, stays a diary command.
- The order of migration steps, as long as every commit leaves the CLI working and the tests green.
- Updating `docs/PLAN.md` with this as a new milestone and adding a `docs/NOTES.md` section on what it teaches.
