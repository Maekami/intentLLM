from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from intent_metrics.aitr import OpenRouterAITRJudge
from intent_metrics.cache import AITRCache
from intent_metrics.config import AITREnvironmentSettings, load_aitr_config
from intent_metrics.evaluator import evaluate_runs
from intent_metrics.report import write_evaluation_outputs
from intent_metrics.traces import TraceLoader, discover_run_dirs

app = typer.Typer(add_completion=False, help="Evaluate completed interaction-pipeline runs.")
console = Console()


@app.command()
def evaluate(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="One run directory or a batch directory containing runs.",
    ),
    dataset: Path | None = typer.Option(
        None,
        "--dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Override simulator.dataset_path from each config snapshot.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Result directory; defaults to INPUT_DIR/metrics_results.",
    ),
    cache_path: Path | None = typer.Option(
        None,
        "--cache",
        help="AITR cache path; defaults to OUTPUT_DIR/aitr_cache.json.",
    ),
    skip_aitr: bool = typer.Option(
        False,
        "--skip-aitr",
        help="Compute only deterministic metrics without an external judge call.",
    ),
    aitr_config: Path | None = typer.Option(
        None,
        "--aitr-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Custom AITR judge YAML merged over configs/aitr.yaml.",
    ),
    aitr_model: str | None = typer.Option(
        None,
        "--aitr-model",
        help="Override the AITR YAML model_id and AITR_MODEL.",
    ),
    aitr_reasoning_effort: str | None = typer.Option(
        None,
        "--aitr-reasoning-effort",
        help="Override the AITR YAML reasoning.effort and environment.",
    ),
) -> None:
    source = input_dir.expanduser().resolve()
    destination = (
        output_dir.expanduser().resolve() if output_dir is not None else source / "metrics_results"
    )
    run_dirs = discover_run_dirs(source)
    loader = TraceLoader(dataset)

    judge = None
    cache = None
    if not skip_aitr:
        overrides: dict = {}
        if aitr_model is not None:
            overrides["model_id"] = aitr_model
        if aitr_reasoning_effort is not None:
            overrides["reasoning"] = {
                "enabled": aitr_reasoning_effort.lower() != "none",
                "effort": aitr_reasoning_effort,
            }
        config = load_aitr_config(aitr_config, cli_overrides=overrides)
        environment = AITREnvironmentSettings()
        if not environment.openrouter_api_key:
            console.print(
                "[red]OPENROUTER_API_KEY is required for AITR. "
                "Use --skip-aitr for the three offline metrics.[/red]"
            )
            raise typer.Exit(2)
        judge = OpenRouterAITRJudge(config, environment)
        resolved_cache = (
            cache_path.expanduser().resolve()
            if cache_path is not None
            else destination / "aitr_cache.json"
        )
        cache = AITRCache(resolved_cache)

    result = asyncio.run(
        evaluate_runs(
            [str(path) for path in run_dirs],
            loader=loader,
            judge=judge,
            cache=cache,
        )
    )
    episode_path, aggregate_path, text_path = write_evaluation_outputs(destination, result)
    console.print(
        f"Evaluated [bold]{len(result.records)}[/bold] episode(s); "
        f"[red]{len(result.failures)} trace error(s)[/red].\n"
        f"Token counts unavailable for "
        f"{sum(record.assistant_tokens is None for record in result.records)} episode(s); "
        "their E/S metrics are retained.\n"
        f"{episode_path}\n{aggregate_path}\n{text_path}"
    )
    has_aitr_errors = any(record.aitr_error is not None for record in result.records)
    if result.failures or has_aitr_errors:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
