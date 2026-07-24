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
)
from user_simulator.data.loader import DatasetLoader
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.factory import SimulatorComponents, build_simulator_components
from user_simulator.llm.schema_registry import SCHEMA_REGISTRY

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
    controller_prompt: Path | None = typer.Option(
        None, "--controller-prompt", help="Override controller prompt YAML."
    ),
    satisfaction_prompt: Path | None = typer.Option(
        None, "--satisfaction-prompt", help="Override satisfaction prompt YAML."
    ),
    clear_prompt: Path | None = typer.Option(
        None, "--clear-prompt", help="Override clear-realizer prompt YAML."
    ),
    abstract_prompt: Path | None = typer.Option(
        None, "--abstract-prompt", help="Override abstract-realizer prompt YAML."
    ),
    controller_component: str | None = typer.Option(
        None, "--controller-component", help="Override controller registry name."
    ),
    satisfaction_component: str | None = typer.Option(
        None, "--satisfaction-component", help="Override satisfaction registry name."
    ),
    realizer_component: str | None = typer.Option(
        None, "--realizer-component", help="Override user-realizer registry name."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Run deterministic offline components without an API key."
    ),
    show_latent_summary: bool = typer.Option(
        False,
        "--show-latent-summary",
        help="Show latent task summary in the audit-only startup panel.",
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
    prompt_overrides = {
        "controller": controller_prompt,
        "satisfaction": satisfaction_prompt,
        "realizer_clear": clear_prompt,
        "realizer_abstract": abstract_prompt,
    }
    for key, value in prompt_overrides.items():
        if value is not None:
            overrides.setdefault("prompts", {})[key] = str(value)
    component_overrides = {
        "controller": controller_component,
        "satisfaction_updater": satisfaction_component,
        "user_realizer": realizer_component,
    }
    for key, value in component_overrides.items():
        if value is not None:
            overrides.setdefault("components", {})[key] = value
    if model_profile:
        overrides["models"] = {
            key: model_profile
            for key in (
                "controller",
                "satisfaction",
                "realizer_clear",
                "realizer_abstract",
            )
        }
    if mock:
        overrides.setdefault("components", {}).update(
            {
                "controller": "mock_controller",
                "satisfaction_updater": "mock_satisfaction_updater",
                "user_realizer": "mock_user_realizer",
            }
        )
    resolved = load_config(config, cli_overrides=overrides)
    environment = EnvironmentSettings()
    uses_live_llm = any(
        name.startswith("llm_")
        for name in (
            resolved.components.controller,
            resolved.components.satisfaction_updater,
            resolved.components.user_realizer,
        )
    )
    if uses_live_llm and not environment.openrouter_api_key:
        console.print("[red]OPENROUTER_API_KEY is required for the live demo.[/red]")
        raise typer.Exit(2)
    loader = DatasetLoader(resolved.dataset_path)
    sample = loader.load_sample_by_id(sample_id) if sample_id else loader.sample_randomly(seed)
    components = build_simulator_components(
        config=resolved,
        environment=environment,
    )
    audit = AuditLogger(
        sample.sample_id,
        output_dir=resolved.audit.output_dir,
        level=resolved.audit.level,
        enabled=resolved.audit.enabled,
        config_snapshot={
            "simulator": resolved.model_dump(mode="json"),
            "model_profiles": {
                key: value.model_dump(mode="json")
                for key, value in components.model_profiles.items()
            },
            "difficulty": difficulty.value,
            "seed": seed,
        },
    )
    episode = Episode(
        sample=sample,
        difficulty=difficulty,
        seed=seed,
        controller=components.controller,
        satisfaction_updater=components.satisfaction_updater,
        selection_policy=components.selection_policy,
        realization_policy=components.realization_policy,
        user_realizer=components.user_realizer,
        audit_logger=audit,
        max_turns=resolved.policy.max_turns,
        monotonic_satisfaction=resolved.policy.monotonic_satisfaction,
        auto_expose_backbone_on_empty_queue=(resolved.policy.auto_expose_backbone_on_empty_queue),
    )
    asyncio.run(_run_loop(episode, components, show_latent_summary))


async def _run_loop(
    episode: Episode,
    components: SimulatorComponents,
    show_latent_summary: bool = False,
) -> None:
    renderer = ConsoleAuditRenderer(console)
    sample = episode.sample
    startup = (
        f"Sample: {sample.sample_id}\n"
        "Latent information is hidden; use audit-only commands explicitly."
    )
    if show_latent_summary:
        startup += f"\n[AUDIT-ONLY] Task summary: {sample.task_summary}"
    console.print(Panel(startup, title="Reason-DAG user simulator"))
    initial = await episode.start()
    console.print(f"\n[bold cyan]User:[/bold cyan] {initial.user_message}")
    while not episode.state.terminated:
        response = console.input("\n[bold green]Assistant:[/bold green] ")
        if response.startswith("/"):
            if _handle_command(response.strip(), episode, renderer, components):
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


def _handle_command(
    command: str,
    episode: Episode,
    renderer: ConsoleAuditRenderer,
    components: SimulatorComponents,
) -> bool:
    if command == "/audit":
        renderer.latest(episode.audit)
    elif command == "/raw-audit":
        renderer.raw_latest(episode.audit)
    elif command == "/prompts":
        rendered_hashes = {}
        for event in episode.audit.events:
            call = event.payload.get("llm_call", {})
            if call.get("prompt_name"):
                rendered_hashes[call["prompt_name"]] = call.get("prompt_hash")
        console.print(
            Pretty(
                {
                    key: {
                        "name": prompt.name,
                        "version": prompt.version,
                        "path": prompt.path,
                        "schema": f"{prompt.schema_name}@{prompt.schema_version}",
                        "latest_rendered_hash": rendered_hashes.get(prompt.name),
                    }
                    for key, prompt in components.prompts.items()
                }
            )
        )
    elif command == "/schemas":
        console.print(
            Pretty(
                {
                    f"{name}@{version}": spec.hash
                    for (name, version), spec in SCHEMA_REGISTRY.items()
                }
            )
        )
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
        console.print(
            "Commands: /audit /raw-audit /prompts /schemas /state /dag /nodes /history /save /quit"
        )
    return False


if __name__ == "__main__":
    app()
