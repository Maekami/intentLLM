from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from interaction_pipeline.core import RunResult
from interaction_pipeline.retry_failed import (
    load_failed_batch_source,
    prepare_pipeline_from_batch_snapshot,
    retry_failed_batch,
)

app = typer.Typer(
    add_completion=False,
    help="Retry only failed samples from a frozen evaluation batch.",
)
console = Console()


@app.command()
def main(
    batch_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Existing batch directory containing batch_summary.json and batch_config.yaml.",
    ),
    max_additional_attempts: int = typer.Option(
        3,
        "--max-additional-attempts",
        min=1,
        help="Maximum new attempts for each existing failed sample (default: 3).",
    ),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        min=1,
        help="Concurrent failed samples; defaults to the source batch concurrency.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Recovery output root; defaults to SOURCE_BATCH/repairs.",
    ),
    infrastructure_failures_only: bool = typer.Option(
        False,
        "--infrastructure-failures-only",
        help="Select only final infrastructure failures; leave behavioral failures unchanged.",
    ),
    show_progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    try:
        source = load_failed_batch_source(
            batch_dir,
            infrastructure_failures_only=infrastructure_failures_only,
        )
        if not source.failed_runs:
            selection = (
                "infrastructure failures"
                if infrastructure_failures_only
                else "failed samples"
            )
            console.print(f"[green]Batch has no {selection}:[/green] {source.directory}")
            return
        prepared = prepare_pipeline_from_batch_snapshot(source)
    except (OSError, TypeError, ValueError) as exc:
        console.print(f"[red]Cannot prepare failed-sample recovery: {exc}[/red]")
        raise typer.Exit(2) from exc
    if prepared.missing_credentials:
        console.print(
            "[red]Missing required credentials: "
            + ", ".join(prepared.missing_credentials)
            + ".[/red]"
        )
        raise typer.Exit(2)
    effective_concurrency = concurrency or int(source.summary.get("concurrency", 1))
    options = {
        "output_root": output_dir,
        "concurrency": effective_concurrency,
        "max_additional_attempts": max_additional_attempts,
    }
    if show_progress:
        counts = {"succeeded": 0, "failed": 0}
        with Progress(
            SpinnerColumn(),
            TextColumn("Retrying failed samples"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn(
                "[green]ok {task.fields[succeeded]}[/green] [red]failed {task.fields[failed]}[/red]"
            ),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "Retrying failed samples",
                total=len(source.failed_runs),
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

            repair_dir, summary = asyncio.run(
                retry_failed_batch(
                    prepared,
                    source,
                    **options,
                    progress_callback=advance,
                )
            )
    else:
        repair_dir, summary = asyncio.run(retry_failed_batch(prepared, source, **options))
    recovery = summary["recovery"]
    console.print(
        f"Recovery complete: [green]{recovery['recovered']} recovered[/green], "
        f"[red]{recovery['still_failed']} selected samples still failed[/red], "
        f"{recovery['remaining_batch_failures']} total batch failures remain\n"
        f"Consolidated {summary['sample_count']}-sample batch:\n{repair_dir}"
    )


if __name__ == "__main__":
    app()
