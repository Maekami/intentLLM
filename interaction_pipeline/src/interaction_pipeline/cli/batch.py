import asyncio
from pathlib import Path

import typer
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

from interaction_pipeline.batch import run_batch
from interaction_pipeline.cli.common import configure
from interaction_pipeline.core import RunResult

app = typer.Typer(add_completion=False, help="Run simulator/assistant samples concurrently.")
console = Console()


@app.command()
def main(
    sample_id: list[str] | None = typer.Option(
        None, "--sample-id", help="Sample ID to run; repeat for multiple samples."
    ),
    all_samples: bool = typer.Option(
        False, "--all", help="Run all dataset samples (may incur substantial API cost)."
    ),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Limit --all to the first N samples."
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", min=1, help="Maximum concurrently active samples."
    ),
    sample_retries: int | None = typer.Option(
        None,
        "--sample-retries",
        min=0,
        help="Maximum full-sample retries after the initial attempt (default: 3).",
    ),
    difficulty: str | None = typer.Option(None, "--difficulty", help="easy, medium, or hard."),
    seed: int | None = typer.Option(None, "--seed", help="Base seed; sample index is added."),
    baseline: str | None = typer.Option(None, "--baseline", help="base or prompt_base."),
    assistant_model_profile: str | None = typer.Option(
        None, "--assistant-model-profile", help="Assistant model profile name or YAML path."
    ),
    simulator_model_profile: str | None = typer.Option(
        None, "--simulator-model-profile", help="Simulator model profile name or YAML path."
    ),
    max_turns: int | None = typer.Option(None, "--max-turns", min=1),
    pipeline_config: Path | None = typer.Option(None, "--pipeline-config"),
    simulator_config: Path | None = typer.Option(None, "--simulator-config"),
    assistant_config: Path | None = typer.Option(None, "--assistant-config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show sample-level completion progress.",
    ),
) -> None:
    if all_samples and sample_id:
        raise typer.BadParameter("choose --all or one/more --sample-id values, not both")
    if not all_samples and not sample_id:
        raise typer.BadParameter("provide one/more --sample-id values or explicitly use --all")
    config, prepared, resolved_output = configure(
        pipeline_config=pipeline_config,
        simulator_config=simulator_config,
        assistant_config=assistant_config,
        simulator_model_profile=simulator_model_profile,
        assistant_model_profile=assistant_model_profile,
        baseline=baseline,
        difficulty=difficulty,
        seed=seed,
        concurrency=concurrency,
        sample_retries=sample_retries,
        output_dir=output_dir,
        max_turns=max_turns,
    )
    if prepared.missing_credentials:
        console.print(
            "[red]Missing required credentials: "
            + ", ".join(prepared.missing_credentials)
            + ".[/red]"
        )
        raise typer.Exit(2)
    if all_samples:
        samples = prepared.dataset.load_all_samples()
        if limit is not None:
            samples = samples[:limit]
    else:
        samples = [prepared.dataset.load_sample_by_id(value) for value in sample_id or []]
    batch_options = {
        "output_root": resolved_output,
        "concurrency": config.run.concurrency,
        "sample_retries": config.run.sample_retries,
    }
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
                "Running samples",
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

            batch_dir, results = asyncio.run(
                run_batch(
                    prepared,
                    samples,
                    **batch_options,
                    progress_callback=advance,
                )
            )
    else:
        batch_dir, results = asyncio.run(
            run_batch(
                prepared,
                samples,
                **batch_options,
            )
        )
    completed = sum(item.status == "completed" for item in results)
    console.print(
        f"Batch complete: [green]{completed} completed[/green], "
        f"[red]{len(results) - completed} failed[/red]\n{batch_dir}"
    )


if __name__ == "__main__":
    app()
