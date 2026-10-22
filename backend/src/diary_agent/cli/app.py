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
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import vector_repository as vector_repo
from diary_agent.services.agent.graph import (
    build_graph,
    open_checkpointer,
    pending_interrupt,
    stream_turn,
    thread_config,
)
from diary_agent.services.agent.state import DiaryContext
from diary_agent.services.extraction import build_extractor, get_extraction_model

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
        run_turn(graph, Command(resume=ask_to_save(payload, ctx.day)), config, ctx)
        return
    notice = graph.get_state(config).values.get("notice")
    if notice:
        console.print(f"[green]{notice}[/green]" if notice == "Saved." else f"[dim]{notice}[/dim]")


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
    )
    graph = build_graph(open_checkpointer())
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
            run_turn(graph, Command(resume=ask_to_save(payload, day)), config, ctx)
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


@app.command()
def version():
    """Show package version."""
    console.print("diary-agent 0.1.0")
