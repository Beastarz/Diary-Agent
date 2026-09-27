import typer
from rich.console import Console

from diary_agent.chat import ChatSession
from diary_agent.config import settings
from diary_agent.llm import get_chat_model

app = typer.Typer(help="Local-first chat journal agent.", no_args_is_help=True)
console = Console()


@app.command()
def chat():
    """Start a journaling conversation. Type 'exit' or press Ctrl+C to quit."""
    session = ChatSession(get_chat_model())
    console.print(f"[dim]model: {settings.chat_model} - type 'exit' to quit[/dim]")
    console.print("[bold magenta]diary[/bold magenta]: Hi! How was your day?")
    while True:
        try:
            user_input = console.input("[bold cyan]you[/bold cyan]: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
        if not user_input:
            continue
        console.print("[bold magenta]diary[/bold magenta]: ", end="")
        for chunk in session.stream(user_input):
            console.print(chunk, end="", markup=False, highlight=False)
        console.print()
    console.print("[dim]Goodbye - see you tomorrow.[/dim]")


@app.command()
def version():
    """Show package version."""
    console.print("diary-agent 0.1.0")
