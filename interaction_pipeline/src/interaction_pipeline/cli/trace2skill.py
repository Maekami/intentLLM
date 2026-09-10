import asyncio
from pathlib import Path
from typing import Any

import typer
from assistant.config import EnvironmentSettings as AssistantEnvironmentSettings
from assistant.config import ModelProfile, load_model_profile
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
from interaction_pipeline.config import load_pipeline_config, resolve_pipeline_path
from interaction_pipeline.evolution_dataset import validate_evolution_dataset
from interaction_pipeline.trace2skill import Trace2SkillError, run_trace2skill_collection
from interaction_pipeline.trace2skill_build import run_trace2skill_build
from interaction_pipeline.trace2skill_config import (
    load_trace2skill_config,
    resolve_trace2skill_output_directory,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Collect a sealed No-Skill visible-trajectory corpus, then independently build "
        "an official-aligned Trace2Skill-Visible-Combined-Parallel-B32 skill."
    ),
)
console = Console()


@app.command("collect")
def collect_command(
    assistant_model_profile: str = typer.Option(
        ...,
        "--assistant-model-profile",
        help="Memory-free *_trace2skill profile whose No-Skill behavior is collected.",
    ),
    simulator_model_profile: str | None = typer.Option(
        None,
        "--simulator-model-profile",
        help="Simulator model profile name or YAML path.",
    ),
    trajectory_concurrency: int | None = typer.Option(
        None,
        "--trajectory-concurrency",
        min=1,
        help="Number of independent No-Skill interactions allowed in flight.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Simulator base seed; the zero-based source dataset index is added.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "--smoke-limit",
        min=1,
        max=999,
        help=(
            "Collect the first N source-ordered samples as an isolated audit trial. "
            "--smoke-limit remains a compatibility alias."
        ),
    ),
    trace2skill_config: Path | None = typer.Option(None, "--trace2skill-config"),
    pipeline_config: Path | None = typer.Option(None, "--pipeline-config"),
    simulator_config: Path | None = typer.Option(None, "--simulator-config"),
    assistant_config: Path | None = typer.Option(None, "--assistant-config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        help="Existing trajectory collection directory with matching rollout inputs.",
    ),
    retry_exhausted: bool = typer.Option(
        False,
        "--retry-exhausted",
        help=(
            "With --resume, grant exhausted infrastructure-failure slots one audited "
            "additional retry batch."
        ),
    ),
    show_progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Collect visible trajectories with base (No Skill, no prompted base)."""

    if retry_exhausted and resume is None:
        console.print("[red]--retry-exhausted requires --resume <collection-dir>.[/red]")
        raise typer.Exit(2)

    overrides: dict[str, Any] = {"run": {}}
    for key, value in {
        "trajectory_concurrency": trajectory_concurrency,
        "seed": seed,
        "output_dir": str(output_dir.resolve()) if output_dir else None,
    }.items():
        if value is not None:
            overrides["run"][key] = value
    config = load_trace2skill_config(trace2skill_config, overrides=overrides)
    resolved_dataset = config.dataset.model_copy(
        update={
            key: str(resolve_pipeline_path(getattr(config.dataset, key)))
            for key in ("path", "original_path", "metadata_path")
        }
    )
    base_output = resolve_pipeline_path(config.run.output_dir)
    config = config.model_copy(update={"dataset": resolved_dataset})
    _, prepared, _ = configure(
        pipeline_config=pipeline_config,
        simulator_config=simulator_config,
        assistant_config=assistant_config,
        simulator_model_profile=simulator_model_profile,
        assistant_model_profile=assistant_model_profile,
        baseline="base",
        trace2skill_skill=None,
        difficulty=None,
        seed=config.run.seed,
        concurrency=config.run.trajectory_concurrency,
        sample_retries=0,
        update_memory=False,
        output_dir=base_output,
        max_turns=config.run.turn_budget,
        simulator_dataset_path=resolved_dataset.path,
        assistant_profile_skill_enabled=False,
    )
    profile = load_model_profile(prepared.assistant_config.models["assistant"])
    _require_trace_profile(profile)
    profile_output = resolve_trace2skill_output_directory(
        base_output,
        profile_name=profile.profile_name,
        group_by_profile=config.run.group_output_by_profile,
    )
    collection_output = profile_output / "trajectory_collections"
    if limit is not None:
        collection_output = collection_output / f"trial_n{limit}"
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": str(collection_output)})}
    )
    prepared.config = prepared.config.model_copy(
        update={
            "run": prepared.config.run.model_copy(update={"output_dir": str(collection_output)})
        }
    )
    try:
        samples, dataset_info = validate_evolution_dataset(prepared.dataset, resolved_dataset)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid frozen development dataset: {exc}[/red]")
        raise typer.Exit(2) from exc
    if prepared.missing_credentials:
        console.print(
            "[red]Missing required credentials: "
            + ", ".join(prepared.missing_credentials)
            + ".[/red]"
        )
        raise typer.Exit(2)
    selected_count = limit if limit is not None else len(samples)
    if limit is not None:
        console.print(
            "[yellow]Trace2Skill collection trial[/yellow]: collecting the first "
            f"{limit} source-ordered samples. The sealed trial is audit-only and cannot "
            "be consumed by build or test-set evaluation."
        )

    options = {
        "output_root": collection_output,
        "trace_config": config,
        "dataset_info": dataset_info,
        "resume_dir": resume,
        "limit": limit,
        "retry_exhausted": retry_exhausted,
    }
    if show_progress:
        counts = {"complete": 0, "pending": 0}
        progress, task_id = _progress(
            "Collecting visible trajectories",
            selected_count,
            counts,
            positive="complete",
            negative="pending",
        )
        with progress:

            def advance(finished: int, total: int, slot: dict[str, Any]) -> None:
                key = "complete" if slot.get("status") == "complete" else "pending"
                counts[key] += 1
                progress.update(task_id, completed=finished, total=total, **counts)

            result = asyncio.run(
                run_trace2skill_collection(
                    prepared,
                    samples,
                    **options,
                    progress_callback=advance,
                )
            )
    else:
        result = asyncio.run(run_trace2skill_collection(prepared, samples, **options))

    if not result.complete:
        recovery = (
            f"Resume with --resume {result.run_dir} --retry-exhausted"
            if result.exhausted_pending_count
            else f"Resume with --resume {result.run_dir}"
        )
        console.print(
            f"[red]Trajectory collection incomplete[/red]: "
            f"{result.valid_trajectory_count} complete, {result.pending_count} pending.\n"
            f"{recovery}. Build remains blocked until corpus.json is sealed."
        )
        raise typer.Exit(1)
    if result.canonical:
        console.print(
            f"[green]Trajectory corpus sealed[/green]: {result.valid_trajectory_count} "
            f"trajectories, zero distillation calls.\nCorpus SHA-256: "
            f"{result.corpus_sha256}\nCollection: {result.run_dir}"
        )
    else:
        console.print(
            f"[green]Trajectory collection trial sealed[/green]: "
            f"{result.valid_trajectory_count} trajectories. This corpus is audit-only.\n"
            f"Corpus SHA-256: {result.corpus_sha256}\nCollection: {result.run_dir}"
        )


@app.command("build")
def build_command(
    assistant_model_profile: str = typer.Option(
        ...,
        "--assistant-model-profile",
        help="The same model behavior profile recorded by the trajectory collection.",
    ),
    trajectory_run: Path = typer.Option(
        ...,
        "--trajectory-run",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="A completed and sealed canonical Trace2Skill collect directory.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "--smoke-limit",
        min=1,
        max=999,
        help="Build from the first N trajectories as a retained, non-published audit trial.",
    ),
    analysis_concurrency: int | None = typer.Option(
        None,
        "--analysis-concurrency",
        "--trajectory-concurrency",
        min=1,
        help="Number of independent analysis and MAP slots allowed in flight.",
    ),
    merge_concurrency: int | None = typer.Option(
        None,
        "--merge-concurrency",
        min=1,
        max=32,
    ),
    trace2skill_config: Path | None = typer.Option(None, "--trace2skill-config"),
    pipeline_config: Path | None = typer.Option(
        None,
        "--pipeline-config",
        help="Used only to resolve the Assistant project root; no Simulator is loaded.",
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        help="Official-aligned build directory with identical immutable inputs.",
    ),
    show_progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Build from a sealed corpus without running any new interactions."""

    overrides: dict[str, Any] = {"run": {}}
    for key, value in {
        "trajectory_concurrency": analysis_concurrency,
        "merge_concurrency": merge_concurrency,
        "output_dir": str(output_dir.resolve()) if output_dir else None,
    }.items():
        if value is not None:
            overrides["run"][key] = value
    config = load_trace2skill_config(trace2skill_config, overrides=overrides)
    profile_path = _resolve_assistant_model_profile_path(
        assistant_model_profile,
        pipeline_config=pipeline_config,
    )
    profile = load_model_profile(str(profile_path))
    _require_trace_profile(profile)
    environment = AssistantEnvironmentSettings()
    if profile.provider == "openrouter" and not environment.openrouter_api_key:
        console.print("[red]Missing required credential: OPENROUTER_API_KEY (assistant).[/red]")
        raise typer.Exit(2)
    base_output = resolve_pipeline_path(config.run.output_dir)
    profile_output = resolve_trace2skill_output_directory(
        base_output,
        profile_name=profile.profile_name,
        group_by_profile=config.run.group_output_by_profile,
    )
    build_output = (
        profile_output / "skill_builds" / (f"trial_n{limit}" if limit is not None else "canonical")
    )
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": str(build_output)})}
    )
    selected_count = limit if limit is not None else config.dataset.expected_sample_count
    if limit is not None:
        console.print(
            "[yellow]Trace2Skill build trial[/yellow]: using the first "
            f"{limit} source-ordered trajectories. Every build-stage artifact is retained, "
            "but the skill is not published and cannot be evaluated on the test set."
        )
    options = {
        "collection_dir": trajectory_run,
        "output_root": build_output,
        "trace_config": config,
        "profile_path": profile_path,
        "assistant_environment": environment,
        "resume_dir": resume,
        "limit": limit,
    }
    try:
        if show_progress:
            counts = {"complete": 0, "skipped": 0}
            progress, task_id = _progress(
                "Analyzing trajectories and producing MAP patches",
                selected_count,
                counts,
                positive="complete",
                negative="skipped",
            )
            with progress:

                def advance(finished: int, total: int, slot: dict[str, Any]) -> None:
                    key = "complete" if slot.get("status") == "complete" else "skipped"
                    counts[key] += 1
                    progress.update(task_id, completed=finished, total=total, **counts)

                result = asyncio.run(run_trace2skill_build(**options, progress_callback=advance))
        else:
            result = asyncio.run(run_trace2skill_build(**options))
    except (Trace2SkillError, TypeError, ValueError) as exc:
        console.print(f"[red]Trace2Skill build rejected[/red]: {exc}")
        raise typer.Exit(2) from exc

    if not result.complete:
        console.print(
            f"[red]Trace2Skill build incomplete[/red]: {result.stage_error}. "
            f"No skill was published.\nRun: {result.run_dir}"
        )
        raise typer.Exit(1)
    publication = (
        f"\nPublished profile skill: {result.published_skill_path}" if result.canonical else ""
    )
    mode = "Canonical" if result.canonical else "Trial"
    console.print(
        f"[green]{mode} Trace2Skill build complete[/green]: "
        f"{result.valid_trajectory_count} trajectories, {result.patch_count} MAP patches, "
        f"{result.skipped_count} excluded MAP slots.\nRun-local skill: "
        f"{result.final_skill_path}{publication}\nRun: {result.run_dir}"
    )


def _require_trace_profile(profile: ModelProfile) -> None:
    if profile.memory is not None:
        console.print("[red]Trace2Skill requires a memory-free target model profile.[/red]")
        raise typer.Exit(2)
    if profile.skill is None or profile.skill.framework != "trace2skill":
        console.print(
            "[red]Trace2Skill requires a profile with its own "
            "skill.framework=trace2skill binding.[/red]"
        )
        raise typer.Exit(2)


def _resolve_assistant_model_profile_path(
    value: str,
    *,
    pipeline_config: Path | None,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    pipeline = load_pipeline_config(pipeline_config)
    assistant_root = resolve_pipeline_path(pipeline.projects.assistant_root)
    direct = (assistant_root / candidate).resolve()
    if direct.is_file():
        return direct
    filename = candidate.name if candidate.suffix in {".yaml", ".yml"} else f"{value}.yaml"
    standard = (assistant_root / "configs/models" / filename).resolve()
    if not standard.is_file():
        raise typer.BadParameter(f"assistant model profile does not exist: {value}")
    return standard


def _progress(
    description: str,
    total: int,
    counts: dict[str, int],
    *,
    positive: str,
    negative: str,
) -> tuple[Progress, int]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(
            f"[green]{positive} {{task.fields[{positive}]}}[/green] "
            f"[yellow]{negative} {{task.fields[{negative}]}}[/yellow]"
        ),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task(description, total=total, **counts)
    return progress, task_id


if __name__ == "__main__":
    app()
