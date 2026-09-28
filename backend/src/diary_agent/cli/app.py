from datetime import date

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from diary_agent.core.config import settings
from diary_agent.core.llm import get_chat_model
from diary_agent.core.prompts import PROMPT_VERSION
from diary_agent.models.entry import ExtractedEntry, JournalEntry
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.services.chat import ChatSession
from diary_agent.services.extraction import (
    ExtractionError,
    build_extractor,
    extract_entry,
    format_transcript,
    get_extraction_model,
)

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


def save_flow(session: ChatSession) -> None:
    """Extract -> show -> confirm -> save. The human approves every write."""
    today = date.today()
    transcript = format_transcript(session.history)
    with console.status("Writing up your entry..."):
        try:
            extracted = extract_entry(build_extractor(get_extraction_model()), transcript)
        except ExtractionError as e:
            console.print(f"[red]{e}[/red]")
            return
    console.print(render_entry(extracted, today))

    conn = entry_repo.connect()
    if entry_repo.get_entry(conn, today) and not Confirm.ask("An entry for today already exists. Overwrite it?"):
        return
    if not Confirm.ask("Save this entry?"):
        return
    entry = JournalEntry(
        **extracted.model_dump(),
        entry_date=today,
        transcript=transcript,
        model=settings.chat_model,
        prompt_version=PROMPT_VERSION,
    )
    entry_repo.save_entry(conn, entry)
    session.mark_saved()
    console.print("[green]Saved.[/green]")


@app.command()
def chat():
    """Start a journaling conversation. '/save' writes today's entry; 'exit' quits."""
    session = ChatSession(get_chat_model())
    console.print(f"[dim]model: {settings.chat_model} - '/save' to save the entry, 'exit' to quit[/dim]")
    console.print("[bold magenta]diary[/bold magenta]: Hi! How was your day?")
    while True:
        try:
            # lstrip BOM: PowerShell prepends U+FEFF when piping text into a program.
            user_input = console.input("[bold cyan]you[/bold cyan]: ").lstrip("﻿").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
        if user_input.lower() == "/save":
            if session.history:
                save_flow(session)
            else:
                console.print("[dim]Nothing to save yet - tell me about your day first.[/dim]")
            continue
        if not user_input:
            continue
        console.print("[bold magenta]diary[/bold magenta]: ", end="")
        for chunk in session.stream(user_input):
            console.print(chunk, end="", markup=False, highlight=False)
        console.print()
    try:
        if session.has_unsaved and Confirm.ask("Save today's entry before leaving?"):
            save_flow(session)
    except (KeyboardInterrupt, EOFError):
        pass
    console.print("[dim]Goodbye - see you tomorrow.[/dim]")


@app.command("list")
def list_cmd(limit: int = typer.Option(10, help="How many recent entries to show.")):
    """List recent entries."""
    entries = entry_repo.list_entries(entry_repo.connect(), limit=limit)
    if not entries:
        console.print("[dim]No entries yet. Run 'diary chat' and type /save.[/dim]")
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
def version():
    """Show package version."""
    console.print("diary-agent 0.1.0")
