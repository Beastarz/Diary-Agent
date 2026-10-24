import time
from datetime import date

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from langgraph.types import Command

from diary_agent.core.config import settings
from diary_agent.core.llm import get_chat_model
from diary_agent.models.entry import ExtractedEntry
from diary_agent.models.memory import Memory, MemoryKind
from diary_agent.models.summary import SummaryRecord
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import memory_repository as memory_repo
from diary_agent.repositories import summary_repository as summary_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services import memory as memory_service
from diary_agent.services import summary as summary_service
from diary_agent.services.agent.graph import (
    build_graph,
    open_checkpointer,
    pending_interrupt,
    stream_turn,
    thread_config,
)
from diary_agent.services.agent.state import DiaryContext
from diary_agent.services.extraction import ExtractionError, build_extractor, get_extraction_model

app = typer.Typer(help="Local-first chat journal agent.", no_args_is_help=True)
console = Console()


def render_entry(entry: ExtractedEntry, day: date) -> Panel:
    def items(xs: list[str]) -> str:
        return ", ".join(xs) if xs else "[dim]-[/dim]"

    mood = f"{entry.mood}/10" if entry.mood else "?"
    if entry.mood_words:
        mood += f" ({', '.join(entry.mood_words)})"
    body = (
        f"[bold]Highlights:[/bold] {items(entry.highlights)}\n"
        f"[bold]Challenges:[/bold] {items(entry.challenges)}\n"
        f"[bold]Gratitude:[/bold]  {items(entry.gratitude)}\n"
        f"[bold]Tags:[/bold]       {items(entry.tags)}\n\n"
        f"{entry.summary}"
    )
    return Panel(body, title=f"{day.isoformat()} - mood {mood}", title_align="left", expand=False)


def ask_to_save(payload: dict, day: date) -> bool:
    """Answers the graph's interrupt(): show the draft, get a yes/no from the human."""
    console.print(render_entry(ExtractedEntry.model_validate(payload["draft"]), day))
    if payload["exists"] and not Confirm.ask("An entry for today already exists. Overwrite it?"):
        return False
    return Confirm.ask("Save this entry?")


def ask_to_remember(payload: dict) -> list[bool]:
    """Answers confirm_memory's interrupt(): one yes/no per proposed memory."""
    answers = []
    for p in payload["proposals"]:
        m = p["memory"]
        if p["action"] == "update":
            console.print(f"[bold]Update memory[/bold] ({m['kind']}): {p['replaces']} [dim]->[/dim] {m['content']}")
        else:
            console.print(f"[bold]New memory[/bold] ({m['kind']}): {m['content']}")
        answers.append(Confirm.ask("Remember this?"))
    return answers


def answer_interrupt(payload: dict, day: date) -> bool | list[bool]:
    return ask_to_remember(payload) if "proposals" in payload else ask_to_save(payload, day)


def run_turn(graph, graph_input, config: dict, ctx: DiaryContext) -> None:
    """One graph run: stream the reply, answer any interrupt, show the notice."""
    started = False
    for kind, text in stream_turn(graph, graph_input, config, ctx):
        if kind == "status":
            console.print(f"[dim]{text}[/dim]")
            continue
        if not started:
            console.print("[bold magenta]diary[/bold magenta]: ", end="")
            started = True
        console.print(text, end="", markup=False, highlight=False)
    if started:
        console.print()
    payload = pending_interrupt(graph, config)
    if payload is not None:
        # Ctrl+C here leaves the graph paused; the next `diary chat` picks the question up again.
        run_turn(graph, Command(resume=answer_interrupt(payload, ctx.day)), config, ctx)
        return
    notice = graph.get_state(config).values.get("notice")
    if notice:
        good = notice.startswith(("Saved.", "Remembered")) and "Couldn't" not in notice
        console.print(f"[green]{notice}[/green]" if good else f"[dim]{notice}[/dim]")


@app.command()
def chat():
    """Journal with the agent. Today's conversation is saved as you go and resumes on restart."""
    day = date.today()
    ctx = DiaryContext(
        chat_model=get_chat_model(),
        extractor=build_extractor(get_extraction_model()),
        conn=entry_repo.connect(),
        vectors=vector_repo.open_store(),
        day=day,
        model_name=settings.chat_model,
        memory_extractor=memory_service.build_memory_extractor(get_extraction_model()),
    )
    graph = build_graph(open_checkpointer(), memory_repo.open_store())
    config = thread_config(day)
    console.print(
        f"[dim]model: {settings.chat_model} - /checkin guided questions, /save write the entry, "
        "/cancel stop a check-in, exit quit[/dim]"
    )

    messages = graph.get_state(config).values.get("messages", [])
    if messages:
        console.print(f"[dim]Resuming today's conversation ({len(messages)} messages).[/dim]")
        if messages[-1].type == "ai":  # only repeat a question that's still waiting for an answer
            console.print("[bold magenta]diary[/bold magenta]: ", end="")
            console.print(messages[-1].content, markup=False, highlight=False)
    else:
        console.print("[bold magenta]diary[/bold magenta]: Hi! How was your day?")

    try:
        if (payload := pending_interrupt(graph, config)) is not None:
            run_turn(graph, Command(resume=answer_interrupt(payload, day)), config, ctx)
        while True:
            # lstrip BOM: PowerShell prepends U+FEFF when piping text into a program.
            user_input = console.input("[bold cyan]you[/bold cyan]: ").lstrip("﻿").strip()
            if user_input.lower() in {"exit", "quit", "bye"}:
                break
            if user_input:
                run_turn(graph, {"user_input": user_input}, config, ctx)
    except (KeyboardInterrupt, EOFError):
        pass

    try:
        values = graph.get_state(config).values
        unsaved = len(values.get("messages", [])) > values.get("saved_count", 0)
        if unsaved and Confirm.ask("Save today's entry before leaving?"):
            run_turn(graph, {"user_input": "/save"}, config, ctx)
    except (KeyboardInterrupt, EOFError):
        pass
    console.print("[dim]Goodbye - see you tomorrow.[/dim]")


@app.command("list")
def list_cmd(limit: int = typer.Option(10, help="How many recent entries to show.")):
    """List recent entries."""
    entries = entry_repo.list_entries(entry_repo.connect(), limit=limit)
    if not entries:
        console.print("[dim]No entries yet. Run 'diary chat' and type /checkin or /save.[/dim]")
        return
    table = Table("date", "mood", "tags", "summary")
    for e in entries:
        table.add_row(e.entry_date.isoformat(), str(e.mood or "?"), ", ".join(e.tags), e.summary)
    console.print(table)


@app.command()
def show(day: str = typer.Argument(None, help="YYYY-MM-DD (default: today)"),
         transcript: bool = typer.Option(False, "--transcript", help="Also print the conversation.")):
    """Show one entry."""
    target = date.fromisoformat(day) if day else date.today()
    entry = entry_repo.get_entry(entry_repo.connect(), target)
    if not entry:
        console.print(f"[dim]No entry for {target.isoformat()}.[/dim]")
        raise typer.Exit(1)
    console.print(render_entry(entry, entry.entry_date))
    if transcript:
        console.print(entry.transcript, markup=False, highlight=False)


@app.command()
def reindex():
    """Rebuild the search index from saved entries (e.g. entries saved before M4, or while Ollama was down)."""
    entries = entry_repo.list_entries(entry_repo.connect())
    store = vector_repo.open_store()
    with console.status(f"Embedding {len(entries)} entries..."):
        for entry in entries:
            vector_repo.upsert_entry(store, entry)
    console.print(f"[green]Indexed {len(entries)} entries.[/green] Index now holds {vector_repo.count(store)}.")


def render_summary(s: SummaryRecord) -> Panel:
    def items(xs: list[str]) -> str:
        return "\n".join(f"  - {x}" for x in xs) if xs else "  [dim]-[/dim]"

    mood = f"avg mood {s.mood_avg}/10" if s.mood_avg is not None else "no mood ratings"
    body = (
        f"{s.overview}\n\n"
        f"[bold]Mood:[/bold] {s.mood_trend}\n\n"
        f"[bold]Themes:[/bold]\n{items(s.themes)}\n"
        f"[bold]Wins:[/bold]\n{items(s.wins)}\n"
        f"[bold]Challenges:[/bold]\n{items(s.challenges)}"
    )
    count = f"{s.entry_count} {'entry' if s.entry_count == 1 else 'entries'}"
    title = f"{s.period} {s.start_date.isoformat()} to {s.end_date.isoformat()} - {count}, {mood}"
    return Panel(body, title=title, title_align="left", expand=False)


@app.command()
def summary(
    week: bool = typer.Option(True, "--week/--month", help="Summarize a calendar week (Mon-Sun) or month."),
    day: str = typer.Option(None, "--date", help="Any day in the period, YYYY-MM-DD (default: today)."),
    saved: bool = typer.Option(False, "--saved", help="Show the saved summary instead of writing a new one."),
):
    """Summarize a week or month of entries: mood trend, recurring themes, wins."""
    period = "week" if week else "month"
    start, end = summary_service.period_range(period, date.fromisoformat(day) if day else date.today())
    conn = entry_repo.connect()
    existing = summary_repo.get_summary(conn, period, start)
    if saved:
        if not existing:
            console.print(f"[dim]No saved {period} summary starting {start.isoformat()}.[/dim]")
            raise typer.Exit(1)
        console.print(render_summary(existing))
        return

    entries = entry_repo.list_entries(conn, start=start, end=end)
    if not entries:
        console.print(f"[dim]No entries between {start.isoformat()} and {end.isoformat()}.[/dim]")
        return
    writer = summary_service.build_writer(summary_service.get_summary_model(max_tokens=200))
    reducer = summary_service.build_reducer(summary_service.get_summary_model(max_tokens=800))
    started = time.perf_counter()
    with console.status(f"Condensing {len(entries)} entries, then writing the {period} review..."):
        try:
            record = summary_service.summarize_period(
                writer, reducer, entries, period, start, end, settings.chat_model
            )
        except ExtractionError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    console.print(render_summary(record))
    console.print(f"[dim]{time.perf_counter() - started:.1f}s[/dim]")

    if existing and not Confirm.ask(f"A saved summary for this {period} exists. Overwrite it?"):
        return
    if Confirm.ask("Save this summary?"):
        summary_repo.save_summary(conn, record)
        console.print("[green]Saved.[/green]")


memory_app = typer.Typer(help="See and edit what the agent remembers about you.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")


@memory_app.command("list")
def memory_list():
    """List remembered facts, goals, people and preferences."""
    memories = memory_repo.list_memories(memory_repo.open_store())
    if not memories:
        console.print("[dim]Nothing remembered yet. Say 'remember that ...' in 'diary chat'.[/dim]")
        return
    table = Table("id", "kind", "content", "learned")
    for m in memories:
        table.add_row(m.id, m.kind, m.content, m.source_date.isoformat())
    console.print(table)


@memory_app.command("forget")
def memory_forget(memory_id: str = typer.Argument(..., help="The id from 'diary memory list'.")):
    """Delete one memory."""
    if not memory_repo.delete_memory(memory_repo.open_store(), memory_id):
        console.print(f"[dim]No memory with id {memory_id}.[/dim]")
        raise typer.Exit(1)
    console.print("[green]Forgotten.[/green]")


@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(..., help="e.g. 'Training for a marathon in March'"),
    kind: str = typer.Option("fact", help="fact, goal, person or preference."),
):
    """Add a memory yourself. No LLM and no confirmation: you typed it."""
    if kind not in MemoryKind.__args__:
        console.print(f"[red]kind must be one of {', '.join(MemoryKind.__args__)}[/red]")
        raise typer.Exit(1)
    memory = Memory(id=memory_repo.new_id(), kind=kind, content=content, evidence=content, source_date=date.today())
    memory_repo.put_memory(memory_repo.open_store(), memory)
    console.print(f"[green]Remembered[/green] ({memory.id}).")


@app.command()
def version():
    """Show package version."""
    console.print("diary-agent 0.1.0")
