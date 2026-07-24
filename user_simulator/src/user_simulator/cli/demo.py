import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from user_simulator.audit.console_renderer import ConsoleAuditRenderer
from user_simulator.audit.logger import AuditLogger
from user_simulator.config import (
    EnvironmentSettings,
    load_config,
    load_model_profile,
)
from user_simulator.controller.llm_controller import LLMController
from user_simulator.data.loader import DatasetLoader
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy
from user_simulator.realizer.llm_realizer import LLMUserRealizer
from user_simulator.satisfaction.llm_updater import LLMSatisfactionUpdater

app = typer.Typer(add_completion=False, help="Run an interactive reason-DAG episode.")
console = Console()


@app.command()
def main(
    sample_id: str | None = typer.Option(
        None, "--sample-id", help="Exact dataset sample ID to simulate."
    ),
    random_sample: bool = typer.Option(
        False, "--random-sample", help="Choose a reproducible random sample."
    ),
    difficulty: Difficulty = typer.Option(
        Difficulty.MEDIUM, "--difficulty", help="User simulation difficulty."
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for all episode randomness."),
    config: Path | None = typer.Option(
        None, "--config", help="Custom simulator YAML merged over defaults."
    ),
    model_profile: str | None = typer.Option(
        None,
        "--model-profile",
        help="Override every component with one model profile name or YAML path.",
    ),
    audit_level: str | None = typer.Option(
        None, "--audit-level", help="Audit detail: summary or full."
    ),
    max_turns: int | None = typer.Option(
        None, "--max-turns", help="Maximum assistant turns before failure."
    ),
) -> None:
    """Let a human act as the assistant while the simulator acts as the user."""
    if bool(sample_id) == random_sample:
        raise typer.BadParameter("choose exactly one of --sample-id or --random-sample")
    overrides: dict[str, Any] = {}
    if audit_level:
        overrides.setdefault("audit", {})["level"] = audit_level
    if max_turns is not None:
        overrides.setdefault("policy", {})["max_turns"] = max_turns
    resolved = load_config(config, cli_overrides=overrides)
    environment = EnvironmentSettings()
    if not environment.openrouter_api_key:
        console.print("[red]OPENROUTER_API_KEY is required for the live demo.[/red]")
        raise typer.Exit(2)
    loader = DatasetLoader(resolved.dataset_path)
    sample = loader.load_sample_by_id(sample_id) if sample_id else loader.sample_randomly(seed)
    profile_names = (
        {key: model_profile for key in resolved.models} if model_profile else resolved.models
    )
    profiles = {key: load_model_profile(value) for key, value in profile_names.items()}
    audit = AuditLogger(
        sample.sample_id,
        output_dir=resolved.audit.output_dir,
        level=resolved.audit.level,
        enabled=resolved.audit.enabled,
        config_snapshot={
            "simulator": resolved.model_dump(mode="json"),
            "model_profiles": {
                key: value.model_dump(mode="json") for key, value in profiles.items()
            },
            "difficulty": difficulty.value,
            "seed": seed,
        },
    )
    clients = {
        key: OpenRouterStructuredClient(profile, environment) for key, profile in profiles.items()
    }
    controller_profile = profiles["controller"]
    controller = LLMController(
        clients["controller"],
        controller_profile.generation["controller"],
        model_profile_name=controller_profile.profile_name,
        semantic_attempts=controller_profile.retry.max_attempts,
    )
    satisfaction_profile = profiles["satisfaction"]
    updater = LLMSatisfactionUpdater(
        clients["satisfaction"],
        satisfaction_profile.generation["satisfaction"],
        model_profile_name=satisfaction_profile.profile_name,
        semantic_attempts=satisfaction_profile.retry.max_attempts,
    )
    clear_profile = profiles["realizer_clear"]
    abstract_profile = profiles["realizer_abstract"]
    realizer = LLMUserRealizer(
        clients["realizer_clear"],
        clear_profile.generation["realizer_clear"],
        abstract_profile.generation["realizer_abstract"],
        model_profile_name=clear_profile.profile_name,
        abstract_client=clients["realizer_abstract"],
        abstract_model_profile_name=abstract_profile.profile_name,
    )
    episode = Episode(
        sample=sample,
        difficulty=difficulty,
        seed=seed,
        controller=controller,
        satisfaction_updater=updater,
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(resolved.policy.medium_clear_probability),
        user_realizer=realizer,
        audit_logger=audit,
        max_turns=resolved.policy.max_turns,
        monotonic_satisfaction=resolved.policy.monotonic_satisfaction,
        auto_expose_backbone_on_empty_queue=(resolved.policy.auto_expose_backbone_on_empty_queue),
    )
    asyncio.run(_run_loop(episode))


async def _run_loop(episode: Episode) -> None:
    renderer = ConsoleAuditRenderer(console)
    sample = episode.sample
    console.print(
        Panel(
            f"Sample: {sample.sample_id}\n"
            f"Task summary: {sample.task_summary}\n"
            "Latent nodes are hidden; use /dag or /nodes for audit-only views.",
            title="Reason-DAG user simulator",
        )
    )
    initial = await episode.start()
    console.print(f"\n[bold cyan]User:[/bold cyan] {initial.user_message}")
    while not episode.state.terminated:
        response = await asyncio.to_thread(console.input, "\n[bold green]Assistant:[/bold green] ")
        if response.startswith("/"):
            if _handle_command(response.strip(), episode, renderer):
                break
            continue
        result = await episode.submit_assistant(response)
        renderer.latest(episode.audit)
        if result.terminal:
            console.print("\n[bold magenta]Episode terminated successfully.[/bold magenta]")
            break
        console.print(f"\n[bold cyan]User:[/bold cyan] {result.user_message}")
    episode.audit.save_state(episode.state)
    console.print(f"Audit directory: {episode.audit.run_dir}")


def _handle_command(command: str, episode: Episode, renderer: ConsoleAuditRenderer) -> bool:
    if command == "/audit":
        renderer.latest(episode.audit)
    elif command == "/state":
        renderer.state(episode.state)
    elif command == "/dag":
        console.print(
            Panel(
                Pretty(
                    {
                        "warning": "AUDIT-ONLY latent view",
                        "edges": [edge.model_dump() for edge in episode.sample.reason_dag.edges],
                    }
                ),
                title="DAG",
            )
        )
    elif command == "/nodes":
        console.print(
            Panel(
                Pretty(
                    {
                        "warning": "AUDIT-ONLY latent view",
                        "nodes": [
                            episode.navigator.node(node_id).model_dump()
                            for node_id in episode.state.exposed_nodes
                        ],
                    }
                ),
                title="Exposed nodes",
            )
        )
    elif command == "/history":
        console.print(Pretty([item.model_dump() for item in episode.state.conversation_history]))
    elif command == "/save":
        episode.audit.flush()
        console.print("Logs flushed.")
    elif command == "/quit":
        episode.audit.save_state(episode.state)
        return True
    else:
        console.print("Commands: /audit /state /dag /nodes /history /save /quit")
    return False


if __name__ == "__main__":
    app()
