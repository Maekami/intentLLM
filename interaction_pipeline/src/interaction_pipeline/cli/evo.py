import asyncio
from pathlib import Path
from typing import Any

import typer
from assistant.config import load_model_profile
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from interaction_pipeline.cli.common import configure
from interaction_pipeline.config import resolve_pipeline_path
from interaction_pipeline.core import RunResult
from interaction_pipeline.evolution import run_memory_evolution
from interaction_pipeline.evolution_config import (
    load_evolution_config,
    resolve_profile_output_directory,
)
from interaction_pipeline.evolution_dataset import (
    select_samples_in_dataset_order,
    validate_evolution_dataset,
)

app = typer.Typer(add_completion=False, help="Evolve Evo-Memory on development samples.")
console = Console()


@app.command()
def main(
    sample_id: list[str] | None = typer.Option(
        None, "--sample-id", help="Development sample ID; repeat for multiple samples."
    ),
    all_samples: bool = typer.Option(
        False, "--all", help="Run all development samples (may incur substantial API cost)."
    ),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Limit --all to the first N samples."
    ),
    mini_batch_size: int | None = typer.Option(
        None, "--mini-batch-size", min=1, help="Samples sharing one frozen memory snapshot."
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", min=1, help="Maximum active samples inside a mini-batch."
    ),
    sample_retries: int | None = typer.Option(
        None,
        "--sample-retries",
        min=0,
        help=(
            "Full-development-sample infrastructure retries; turn-limit failures are not "
            "retried and evolution.yaml defaults to 0."
        ),
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Simulator base seed; the original zero-based dataset index is added.",
    ),
    baseline: str | None = typer.Option(None, "--baseline", help="base or prompt_base."),
    assistant_model_profile: str | None = typer.Option(
        None,
        "--assistant-model-profile",
        help="Memory-enabled assistant model profile name or YAML path.",
    ),
    simulator_model_profile: str | None = typer.Option(
        None, "--simulator-model-profile", help="Simulator model profile name or YAML path."
    ),
    max_turns: int | None = typer.Option(None, "--max-turns", min=1),
    evo_config: Path | None = typer.Option(
        None, "--evo-config", help="Evolution YAML overriding configs/evolution.yaml."
    ),
    pipeline_config: Path | None = typer.Option(None, "--pipeline-config"),
    simulator_config: Path | None = typer.Option(None, "--simulator-config"),
    assistant_config: Path | None = typer.Option(None, "--assistant-config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show sample-level completion progress after each mini-batch commits.",
    ),
) -> None:
    if all_samples and sample_id:
        raise typer.BadParameter("choose --all or one/more --sample-id values, not both")
    if not all_samples and not sample_id:
        raise typer.BadParameter("provide one/more --sample-id values or explicitly use --all")

    overrides: dict[str, Any] = {"run": {}}
    for key, value in {
        "mini_batch_size": mini_batch_size,
        "concurrency": concurrency,
        "sample_retries": sample_retries,
        "seed": seed,
        "max_turns": max_turns,
        "output_dir": str(output_dir.resolve()) if output_dir else None,
    }.items():
        if value is not None:
            overrides["run"][key] = value
    evolution_config = load_evolution_config(evo_config, overrides=overrides)
    resolved_output_root = resolve_pipeline_path(evolution_config.run.output_dir)
    resolved_dataset = evolution_config.dataset.model_copy(
        update={
            key: str(resolve_pipeline_path(getattr(evolution_config.dataset, key)))
            for key in ("path", "original_path", "metadata_path")
        }
    )
    evolution_config = evolution_config.model_copy(
        update={
            "dataset": resolved_dataset,
            "run": evolution_config.run.model_copy(
                update={"output_dir": str(resolved_output_root)}
            ),
        }
    )
    _, prepared, _ = configure(
        pipeline_config=pipeline_config,
        simulator_config=simulator_config,
        assistant_config=assistant_config,
        simulator_model_profile=simulator_model_profile,
        assistant_model_profile=assistant_model_profile,
        baseline=baseline,
        trace2skill_skill=None,
        difficulty=None,
        seed=evolution_config.run.seed,
        concurrency=evolution_config.run.concurrency,
        sample_retries=evolution_config.run.sample_retries,
        update_memory=True,
        output_dir=resolved_output_root,
        max_turns=evolution_config.run.max_turns,
        simulator_dataset_path=resolved_dataset.path,
    )
    assistant_profile = load_model_profile(prepared.assistant_config.models["assistant"])
    if assistant_profile.memory is None:
        console.print("[red]Evolution requires a memory-enabled assistant profile.[/red]")
        raise typer.Exit(2)
    resolved_output = resolve_profile_output_directory(
        resolved_output_root,
        profile_name=assistant_profile.profile_name,
        memory_framework=assistant_profile.memory.framework,
        group_by_profile=evolution_config.run.group_output_by_profile,
    )
    evolution_config = evolution_config.model_copy(
        update={"run": evolution_config.run.model_copy(update={"output_dir": str(resolved_output)})}
    )
    prepared.config = prepared.config.model_copy(
        update={"run": prepared.config.run.model_copy(update={"output_dir": str(resolved_output)})}
    )
    try:
        all_development_samples, dataset_info = validate_evolution_dataset(
            prepared.dataset,
            resolved_dataset,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid evolution dataset: {exc}[/red]")
        raise typer.Exit(2) from exc
    if prepared.missing_credentials:
        console.print(
            "[red]Missing required credentials: "
            + ", ".join(prepared.missing_credentials)
            + ".[/red]"
        )
        raise typer.Exit(2)
    if all_samples:
        samples = all_development_samples
        if limit is not None:
            samples = samples[:limit]
    else:
        try:
            samples = select_samples_in_dataset_order(
                all_development_samples,
                sample_id or [],
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--sample-id") from exc

    if show_progress:
        counts = {"succeeded": 0, "failed": 0}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn(
                "[green]ok {task.fields[succeeded]}[/green] [red]failed {task.fields[failed]}[/red]"
            ),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "Evolving memory",
                total=len(samples),
                succeeded=0,
                failed=0,
            )

            def advance(finished: int, total: int, result: RunResult) -> None:
                key = "succeeded" if result.status == "completed" else "failed"
                counts[key] += 1
                progress.update(
                    task_id,
                    completed=finished,
                    total=total,
                    succeeded=counts["succeeded"],
                    failed=counts["failed"],
                )

            evolution_dir, results = asyncio.run(
                run_memory_evolution(
                    prepared,
                    samples,
                    output_root=resolved_output,
                    evolution_config=evolution_config,
                    dataset_info=dataset_info,
                    progress_callback=advance,
                )
            )
    else:
        evolution_dir, results = asyncio.run(
            run_memory_evolution(
                prepared,
                samples,
                output_root=resolved_output,
                evolution_config=evolution_config,
                dataset_info=dataset_info,
            )
        )
    completed = sum(item.status == "completed" for item in results)
    console.print(
        f"Evolution complete: [green]{completed} completed[/green], "
        f"[red]{len(results) - completed} failed[/red]\n{evolution_dir}"
    )


if __name__ == "__main__":
    app()
