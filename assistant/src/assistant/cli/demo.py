import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from assistant.config import EnvironmentSettings, load_config
from assistant.exceptions import ModelRequestError
from assistant.factory import build_assistant_components
from assistant.session import AssistantSession

app = typer.Typer(add_completion=False, help="Run an interactive assistant baseline.")
console = Console()


@app.command()
def main(
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Baseline registry name: base or prompt_base.",
    ),
    model_profile: str | None = typer.Option(
        None,
        "--model-profile",
        help="Model profile name or YAML path.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Custom assistant YAML merged over defaults.",
    ),
    message: str | None = typer.Option(
        None,
        "--message",
        help="Send one message and exit instead of starting an interactive session.",
    ),
) -> None:
    overrides: dict[str, Any] = {}
    if baseline:
        overrides.setdefault("components", {})["baseline"] = baseline
    if model_profile:
        overrides.setdefault("models", {})["assistant"] = model_profile
    resolved = load_config(config, cli_overrides=overrides)
    environment = EnvironmentSettings()
    try:
        components = build_assistant_components(config=resolved, environment=environment)
    except ModelRequestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    asyncio.run(_run(components.session, message=message))


async def _run(session: AssistantSession, *, message: str | None = None) -> None:
    console.print(
        Panel(
            f"Baseline: {session.baseline.name}\n"
            f"Memory: {getattr(session, 'memory_framework', None) or 'disabled'}\n"
            "Conversation history is preserved until /reset or /quit.",
            title="Assistant baseline",
        )
    )
    if message is not None:
        response = await session.respond(message)
        console.print(f"[bold cyan]Assistant:[/bold cyan] {response}")
        return
    while True:
        user_input = _read_user_input()
        if user_input.startswith("/"):
            if _handle_command(user_input.strip(), session):
                return
            continue
        response = await session.respond(user_input)
        console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")


def _read_user_input() -> str:
    console.print(
        "\n[bold green]User:[/bold green] "
        "[dim]Enter adds a line; type /send on its own line to submit.[/dim]"
    )
    lines: list[str] = []
    while True:
        line = console.input("[dim]...[/dim] ")
        stripped = line.strip()
        if not lines and stripped.startswith("/") and stripped != "/send":
            return stripped
        if stripped == "/send":
            if lines and any(item.strip() for item in lines):
                return "\n".join(lines)
            console.print("[yellow]Enter a non-empty message before /send.[/yellow]")
            continue
        lines.append(line)


def _handle_command(command: str, session: AssistantSession) -> bool:
    if command == "/history":
        console.print(Pretty([message.model_dump() for message in session.history]))
    elif command == "/reset":
        session.reset()
        console.print("Conversation history reset.")
    elif command == "/quit":
        return True
    else:
        console.print("Commands: /history /reset /quit; /send submits multiline input")
    return False


if __name__ == "__main__":
    app()
