from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml
from user_simulator.domain.dag import Sample

from interaction_pipeline.core import RunResult, record_setup_failure, run_interaction
from interaction_pipeline.prepare import PreparedPipeline

BatchProgressCallback = Callable[[int, int, RunResult], None]
EVALUATION_RETRY_POLICY = "all_failures"
EVOLUTION_RETRY_POLICY = "infrastructure_failures_only"


@dataclass(frozen=True)
class _SampleExecution:
    result: RunResult
    retry_count: int


async def run_batch(
    prepared: PreparedPipeline,
    samples: list[Sample],
    *,
    output_root: str | Path,
    concurrency: int,
    sample_retries: int = 3,
    update_memory: bool = True,
    sample_seeds: list[int] | None = None,
    progress_callback: BatchProgressCallback | None = None,
) -> tuple[Path, list[RunResult]]:
    if sample_retries < 0:
        raise ValueError("sample_retries must be greater than or equal to 0")
    if sample_seeds is not None and len(sample_seeds) != len(samples):
        raise ValueError("sample_seeds must contain exactly one seed per sample")
    resolved_seeds = (
        list(sample_seeds)
        if sample_seeds is not None
        else [prepared.config.run.seed + index for index in range(len(samples))]
    )
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_dir = Path(output_root) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(UTC).isoformat()
    snapshot_builder = getattr(prepared, "batch_config_snapshot", None)
    if callable(snapshot_builder):
        snapshot_options = {
            "batch_id": batch_id,
            "created_at": created_at,
            "samples": samples,
            "concurrency": concurrency,
        }
        if _accepts_keyword(snapshot_builder, "update_memory"):
            snapshot_options["update_memory"] = update_memory
        if _accepts_keyword(snapshot_builder, "retry_policy"):
            snapshot_options["retry_policy"] = EVALUATION_RETRY_POLICY
        batch_config = snapshot_builder(
            **snapshot_options,
        )
    else:
        # Keeps lightweight test doubles and third-party PreparedPipeline-like
        # objects usable while the production implementation writes the full
        # resolved simulator/assistant snapshot.
        batch_config = {
            "run_overview": {
                "difficulty": getattr(prepared.config.run, "difficulty", None),
                "baseline": None,
                "models": {},
            },
            "schema_version": 1,
            "batch": {
                "batch_id": batch_id,
                "created_at": created_at,
                "concurrency": concurrency,
                "sample_retries": sample_retries,
                "update_memory": update_memory,
                "sample_count": len(samples),
                "samples": [
                    {"sample_id": sample.sample_id, "seed": seed}
                    for sample, seed in zip(samples, resolved_seeds, strict=True)
                ],
            },
            "pipeline": (
                prepared.config.model_dump(mode="json")
                if hasattr(prepared.config, "model_dump")
                else {}
            ),
        }
    batch_config.setdefault("batch", {})["sample_retries"] = sample_retries
    batch_config["batch"]["retry_policy"] = EVALUATION_RETRY_POLICY
    batch_config["batch"]["update_memory"] = update_memory
    batch_config["batch"]["samples"] = [
        {"sample_id": sample.sample_id, "seed": seed}
        for sample, seed in zip(samples, resolved_seeds, strict=True)
    ]
    with (batch_dir / "batch_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(batch_config, handle, allow_unicode=True, sort_keys=False)
    semaphore = asyncio.Semaphore(concurrency)
    build_accepts_run_id = _accepts_keyword(prepared.build, "run_id")

    async def run_attempt(index: int, sample: Sample, run_id: str) -> RunResult:
        try:
            build_options = {
                "output_dir": batch_dir,
                "seed": resolved_seeds[index],
            }
            if build_accepts_run_id:
                build_options["run_id"] = run_id
            built = prepared.build(sample, **build_options)
            return await run_interaction(
                episode=built.episode,
                assistant=built.assistant,
                audit=built.audit,
                update_memory=update_memory,
                stop_before_over_budget_generation=True,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one attempt from the batch
            run_dir = batch_dir / run_id
            if run_dir.exists():
                _discard_run_directory(run_dir, batch_dir=batch_dir)
            return record_setup_failure(
                sample_id=sample.sample_id,
                output_dir=batch_dir,
                error=exc,
                run_id=run_id,
                config_snapshot={
                    "pipeline": prepared.config.model_dump(mode="json")
                    if hasattr(prepared.config, "model_dump")
                    else {},
                },
            )

    async def run_one(index: int, sample: Sample) -> _SampleExecution:
        async with semaphore:
            run_id = _sample_run_id(sample.sample_id, index)
            for attempt_number in range(1, sample_retries + 2):
                result = await run_attempt(index, sample, run_id)
                if result.status == "completed" or attempt_number > sample_retries:
                    return _SampleExecution(result=result, retry_count=attempt_number - 1)
                try:
                    _discard_run_directory(result.run_dir, batch_dir=batch_dir)
                except Exception as cleanup_error:  # noqa: BLE001 - preserve sample isolation
                    original_error = result.error or "sample attempt failed"
                    result = replace(
                        result,
                        error=(
                            f"{original_error}; retry cleanup failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        ),
                    )
                    return _SampleExecution(result=result, retry_count=attempt_number - 1)
            raise AssertionError("sample retry loop exited without a result")

    async def run_indexed(index: int, sample: Sample) -> tuple[int, _SampleExecution]:
        return index, await run_one(index, sample)

    tasks = [
        asyncio.create_task(run_indexed(index, sample)) for index, sample in enumerate(samples)
    ]
    ordered_executions: list[_SampleExecution | None] = [None] * len(samples)
    for finished, task in enumerate(asyncio.as_completed(tasks), start=1):
        index, execution = await task
        ordered_executions[index] = execution
        if progress_callback is not None:
            progress_callback(finished, len(samples), execution.result)
    executions = [cast(_SampleExecution, item) for item in ordered_executions]
    results = [execution.result for execution in executions]
    total_retries = sum(item.retry_count for item in executions)
    summary = {
        "batch_id": batch_id,
        "created_at": created_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "concurrency": concurrency,
        "sample_retries": sample_retries,
        "retry_policy": EVALUATION_RETRY_POLICY,
        "update_memory": update_memory,
        "sample_count": len(samples),
        "total_attempts": len(samples) + total_retries,
        "total_retries": total_retries,
        "retried_samples": sum(item.retry_count > 0 for item in executions),
        "completed": sum(item.status == "completed" for item in results),
        "behavioral_failures": sum(is_turn_limit_failure(item) for item in results),
        "infrastructure_failures": sum(
            item.status != "completed" and not is_turn_limit_failure(item) for item in results
        ),
        "failed": sum(item.status != "completed" for item in results),
        "runs": [
            {
                "sample_id": execution.result.sample_id,
                "status": execution.result.status,
                "outcome": batch_result_outcome(execution.result),
                "turns": execution.result.turns,
                "run_dir": str(execution.result.run_dir),
                "error": execution.result.error,
                "retry_count": execution.retry_count,
            }
            for execution in executions
        ],
    }
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"Batch: {batch_id}",
        (
            f"Samples: {len(samples)} | Completed: {summary['completed']} "
            f"| Turn-limit failures: {summary['behavioral_failures']} "
            f"| Infrastructure failures: {summary['infrastructure_failures']}"
        ),
        (
            f"Full-sample retry limit: {sample_retries} | "
            f"Retried samples: {summary['retried_samples']} "
            f"| Retries used: {total_retries} | Total attempts: {summary['total_attempts']}"
        ),
        "",
    ]
    for execution in executions:
        item = execution.result
        lines.append(
            f"[{item.status.upper()}] {item.sample_id} | retries={execution.retry_count} "
            f"| turns={item.turns} | {item.run_dir}" + (f" | {item.error}" if item.error else "")
        )
    (batch_dir / "batch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return batch_dir, results


def _sample_run_id(sample_id: str, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._") or "sample"
    return f"{slug[:90]}_{index + 1:04d}"


def is_turn_limit_failure(result: RunResult) -> bool:
    return (
        result.status != "completed"
        and isinstance(result.error, str)
        and result.error.startswith("EpisodeTurnLimitError:")
    )


def batch_result_outcome(result: RunResult) -> str:
    if result.status == "completed":
        return "SUCCESS"
    if is_turn_limit_failure(result):
        return "FAILURE_TURN_LIMIT"
    return "INFRASTRUCTURE_FAILURE"


def _accepts_keyword(callable_value: object, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_value).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _discard_run_directory(run_dir: Path, *, batch_dir: Path) -> None:
    """Delete one superseded attempt after verifying it is inside this batch."""

    source = run_dir.resolve()
    batch_root = batch_dir.resolve()
    if source.parent != batch_root:
        raise ValueError(f"refusing to delete run directory outside batch root: {source}")
    if not source.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {source}")
    shutil.rmtree(source)
