from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml
from assistant.config import load_model_profile
from assistant.memory.models import PreparedMemoryUpdate
from assistant.memory.sessions import EvolvingMemorySession
from assistant.memory.store import JsonMemoryStore
from user_simulator.audit.logger import AuditLogger
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty

from interaction_pipeline.audit import render_human_audit
from interaction_pipeline.batch import (
    EVOLUTION_RETRY_POLICY,
    BatchProgressCallback,
    batch_result_outcome,
    is_turn_limit_failure,
)
from interaction_pipeline.core import RunResult, record_setup_failure, run_interaction
from interaction_pipeline.evolution_config import EvolutionConfig
from interaction_pipeline.evolution_dataset import (
    EvolutionDatasetInfo,
    sample_evolution_difficulty,
)
from interaction_pipeline.prepare import PreparedPipeline


@dataclass(frozen=True)
class _DeferredUpdate:
    prepared: PreparedMemoryUpdate
    audit: AuditLogger
    turn_index: int


@dataclass(frozen=True)
class _EvolutionExecution:
    result: RunResult
    retry_count: int
    update: _DeferredUpdate | None
    dataset_index: int
    seed: int
    difficulty: Difficulty


async def run_memory_evolution(
    prepared: PreparedPipeline,
    samples: list[Sample],
    *,
    output_root: str | Path,
    evolution_config: EvolutionConfig,
    dataset_info: EvolutionDatasetInfo | None = None,
    progress_callback: BatchProgressCallback | None = None,
) -> tuple[Path, list[RunResult]]:
    """Run ordered mini-batches against frozen memory snapshots."""

    settings = evolution_config.run
    difficulty_field = evolution_config.dataset.difficulty_field
    sample_difficulties = [
        sample_evolution_difficulty(sample, field=difficulty_field) for sample in samples
    ]
    if dataset_info is None:
        dataset_indices = list(range(len(samples)))
    else:
        index_by_sample_id = dataset_info.index_by_sample_id
        try:
            dataset_indices = [index_by_sample_id[sample.sample_id] for sample in samples]
        except KeyError as exc:
            raise ValueError(
                f"sample {exc.args[0]!r} is absent from the validated evolution dataset"
            ) from exc
        if dataset_indices != sorted(dataset_indices):
            raise ValueError("memory evolution samples must follow source JSONL order")
    profile = load_model_profile(prepared.assistant_config.models["assistant"])
    if profile.memory is None:
        raise ValueError("memory evolution requires an assistant profile with memory enabled")
    store = JsonMemoryStore(profile.memory)
    evolution_id = "evo_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    evolution_id += "_" + uuid.uuid4().hex[:8]
    evolution_dir = Path(output_root) / evolution_id
    evolution_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(UTC).isoformat()
    initial_count = len(store.entries())
    _write_evolution_config(
        prepared=prepared,
        samples=samples,
        evolution_config=evolution_config,
        evolution_id=evolution_id,
        evolution_dir=evolution_dir,
        created_at=created_at,
        memory_path=store.path,
        framework=profile.memory.framework,
        initial_count=initial_count,
        max_entries=profile.memory.max_entries,
        dataset_info=dataset_info,
        dataset_indices=dataset_indices,
        sample_difficulties=sample_difficulties,
    )

    semaphore = asyncio.Semaphore(settings.concurrency)
    ordered_executions: list[_EvolutionExecution | None] = [None] * len(samples)
    batch_records: list[dict[str, object]] = []
    finished_count = 0

    for batch_start in range(0, len(samples), settings.mini_batch_size):
        batch_number = batch_start // settings.mini_batch_size + 1
        batch_samples = samples[batch_start : batch_start + settings.mini_batch_size]
        snapshot = tuple(store.entries())

        async def run_attempt(
            sample: Sample,
            run_id: str,
            dataset_index: int,
            difficulty: Difficulty,
            sample_seed: int,
            snapshot=snapshot,
            batch_number: int = batch_number,
        ) -> tuple[RunResult, _DeferredUpdate | None]:
            try:
                built = prepared.build(
                    sample,
                    output_dir=evolution_dir,
                    seed=sample_seed,
                    run_id=run_id,
                    assistant_memory_snapshot=snapshot,
                    difficulty=difficulty,
                    execution_context={
                        "mode": "memory_evolution",
                        "evolution_id": evolution_id,
                        "mini_batch_number": batch_number,
                        "mini_batch_size": settings.mini_batch_size,
                        "concurrency": settings.concurrency,
                        "memory_snapshot_entry_count": len(snapshot),
                        "dataset_index": dataset_index,
                        "difficulty": difficulty.value,
                        "seed": sample_seed,
                    },
                )
                deferred: _DeferredUpdate | None = None

                def defer_update(**values) -> None:
                    nonlocal deferred
                    assistant = values["assistant"]
                    audit = values["audit"]
                    turn_index = values["turn_index"]
                    if not isinstance(assistant, EvolvingMemorySession):
                        raise TypeError(
                            "memory evolution requires an ExpRAGSession or ReMemSession"
                        )
                    candidate = assistant.prepare_memory_update(
                        task_id=values["task_id"],
                        success=values["success"],
                        feedback=values["feedback"],
                    )
                    if candidate.entry is None:
                        result = assistant.memory_store.skipped(
                            task_id=candidate.task_id,
                            reason=candidate.reason,
                        ).to_dict()
                        audit.log("assistant_memory_update_skipped", turn_index, result)
                        return
                    candidate.entry.metadata["evolution"] = {
                        "evolution_id": evolution_id,
                        "dataset_index": dataset_index,
                        "difficulty": difficulty.value,
                        "seed": sample_seed,
                        "dataset_sha256": dataset_info.sha256 if dataset_info is not None else None,
                    }
                    deferred = _DeferredUpdate(
                        prepared=candidate,
                        audit=audit,
                        turn_index=turn_index,
                    )
                    audit.log(
                        "assistant_memory_update_deferred",
                        turn_index,
                        {
                            "task_id": candidate.task_id,
                            "path": str(store.path),
                            "reason": candidate.reason,
                            "mini_batch_number": batch_number,
                        },
                    )

                result = await run_interaction(
                    episode=built.episode,
                    assistant=built.assistant,
                    audit=built.audit,
                    update_memory=True,
                    task_finalizer=defer_update,
                    stop_before_over_budget_generation=True,
                )
                return result, deferred
            except Exception as exc:  # noqa: BLE001 - isolate one dev sample
                run_dir = evolution_dir / run_id
                if run_dir.exists():
                    _discard_run_directory(run_dir, evolution_dir=evolution_dir)
                return (
                    record_setup_failure(
                        sample_id=sample.sample_id,
                        output_dir=evolution_dir,
                        error=exc,
                        run_id=run_id,
                        config_snapshot={
                            "mode": "memory_evolution",
                            "evolution": evolution_config.model_dump(mode="json"),
                            "dataset_index": dataset_index,
                            "difficulty": difficulty.value,
                            "seed": sample_seed,
                        },
                    ),
                    None,
                )

        async def run_one(index: int, sample: Sample) -> _EvolutionExecution:
            async with semaphore:
                dataset_index = dataset_indices[index]
                difficulty = sample_difficulties[index]
                sample_seed = settings.seed + dataset_index
                run_id = _sample_run_id(sample.sample_id, dataset_index)
                for attempt_number in range(1, settings.sample_retries + 2):
                    result, update = await run_attempt(
                        sample,
                        run_id,
                        dataset_index,
                        difficulty,
                        sample_seed,
                    )
                    if (
                        result.status == "completed"
                        or is_turn_limit_failure(result)
                        or attempt_number > settings.sample_retries
                    ):
                        return _EvolutionExecution(
                            result=result,
                            retry_count=attempt_number - 1,
                            update=update,
                            dataset_index=dataset_index,
                            seed=sample_seed,
                            difficulty=difficulty,
                        )
                    _discard_run_directory(result.run_dir, evolution_dir=evolution_dir)
                raise AssertionError("evolution retry loop exited without a result")

        async def run_indexed(
            offset: int,
            sample: Sample,
            batch_start: int = batch_start,
        ) -> tuple[int, _EvolutionExecution]:
            index = batch_start + offset
            return index, await run_one(index, sample)

        tasks = [
            asyncio.create_task(run_indexed(offset, sample))
            for offset, sample in enumerate(batch_samples)
        ]
        batch_executions: list[_EvolutionExecution] = []
        for task in asyncio.as_completed(tasks):
            index, execution = await task
            ordered_executions[index] = execution
        for index in range(batch_start, batch_start + len(batch_samples)):
            batch_executions.append(cast(_EvolutionExecution, ordered_executions[index]))

        ready = [item.update for item in batch_executions if item.update is not None]
        try:
            commit_results = store.upsert_many(
                [item.prepared.entry for item in ready if item.prepared.entry is not None],
                expected_entries=snapshot,
            )
        except Exception as exc:
            for item in ready:
                audit = item.audit
                audit.log(
                    "assistant_memory_update_failed",
                    item.turn_index,
                    {"error_type": type(exc).__name__, "message": str(exc)},
                )
                render_human_audit(audit.run_dir)
            raise

        for item, result in zip(ready, commit_results, strict=True):
            audit = item.audit
            event_type = (
                "assistant_memory_updated" if result.stored else "assistant_memory_update_skipped"
            )
            audit.log(event_type, item.turn_index, result.to_dict())
            render_human_audit(audit.run_dir)

        batch_records.append(
            {
                "mini_batch_number": batch_number,
                "sample_start_index": batch_start,
                "sample_count": len(batch_samples),
                "memory_before": len(snapshot),
                "candidate_count": len(ready),
                "stored_count": sum(result.stored for result in commit_results),
                "memory_after": len(store.entries()),
                "sample_ids": [sample.sample_id for sample in batch_samples],
                "dataset_indices": dataset_indices[batch_start : batch_start + len(batch_samples)],
                "difficulties": [
                    difficulty.value
                    for difficulty in sample_difficulties[
                        batch_start : batch_start + len(batch_samples)
                    ]
                ],
            }
        )
        for execution in batch_executions:
            finished_count += 1
            if progress_callback is not None:
                progress_callback(finished_count, len(samples), execution.result)

    executions = [cast(_EvolutionExecution, item) for item in ordered_executions]
    results = [item.result for item in executions]
    _write_evolution_summary(
        evolution_dir=evolution_dir,
        evolution_id=evolution_id,
        created_at=created_at,
        results=results,
        executions=executions,
        batch_records=batch_records,
        initial_count=initial_count,
        final_count=len(store.entries()),
        memory_path=store.path,
        evolution_config=evolution_config,
        dataset_info=dataset_info,
    )
    return evolution_dir, results


def _write_evolution_config(
    *,
    prepared: PreparedPipeline,
    samples: list[Sample],
    evolution_config: EvolutionConfig,
    evolution_id: str,
    evolution_dir: Path,
    created_at: str,
    memory_path: Path,
    framework: str,
    initial_count: int,
    max_entries: int,
    dataset_info: EvolutionDatasetInfo | None,
    dataset_indices: list[int],
    sample_difficulties: list[Difficulty],
) -> None:
    snapshot = prepared.batch_config_snapshot(
        batch_id=evolution_id,
        created_at=created_at,
        samples=samples,
        concurrency=evolution_config.run.concurrency,
        update_memory=True,
        retry_policy=EVOLUTION_RETRY_POLICY,
    )
    snapshot["evolution"] = {
        **evolution_config.model_dump(mode="json")["run"],
        "update_policy": "frozen_snapshot_then_atomic_batch_commit",
        "commit_order": "dataset_order",
        "difficulty_source": f"dataset.{evolution_config.dataset.difficulty_field}",
        "pipeline_default_difficulty_ignored": getattr(
            prepared.config.run,
            "difficulty",
            None,
        ),
        "memory_path": str(memory_path),
        "memory_framework": framework,
        "memory_initial_entry_count": initial_count,
        "memory_max_entries": max_entries,
    }
    snapshot["run_overview"]["difficulty"] = "per_sample_from_frozen_dataset"
    snapshot["batch"]["mode"] = "memory_evolution"
    snapshot["batch"]["mini_batch_size"] = evolution_config.run.mini_batch_size
    snapshot["batch"]["samples"] = [
        {
            "sample_id": sample.sample_id,
            "dataset_index": dataset_index,
            "difficulty": difficulty.value,
            "seed": evolution_config.run.seed + dataset_index,
        }
        for sample, dataset_index, difficulty in zip(
            samples,
            dataset_indices,
            sample_difficulties,
            strict=True,
        )
    ]
    snapshot["evolution_dataset"] = (
        dataset_info.to_dict()
        if dataset_info is not None
        else evolution_config.dataset.model_dump(mode="json")
    )
    with (evolution_dir / "evolution_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)


def _write_evolution_summary(
    *,
    evolution_dir: Path,
    evolution_id: str,
    created_at: str,
    results: list[RunResult],
    executions: list[_EvolutionExecution],
    batch_records: list[dict[str, object]],
    initial_count: int,
    final_count: int,
    memory_path: Path,
    evolution_config: EvolutionConfig,
    dataset_info: EvolutionDatasetInfo | None,
) -> None:
    total_retries = sum(item.retry_count for item in executions)
    summary = {
        "evolution_id": evolution_id,
        "created_at": created_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "mini_batch_size": evolution_config.run.mini_batch_size,
        "concurrency": evolution_config.run.concurrency,
        "sample_retries": evolution_config.run.sample_retries,
        "retry_policy": EVOLUTION_RETRY_POLICY,
        "sample_count": len(results),
        "completed": sum(item.status == "completed" for item in results),
        "behavioral_failures": sum(is_turn_limit_failure(item) for item in results),
        "infrastructure_failures": sum(
            item.status != "completed" and not is_turn_limit_failure(item) for item in results
        ),
        "failed": sum(item.status != "completed" for item in results),
        "total_retries": total_retries,
        "memory_path": str(memory_path),
        "memory_initial_entry_count": initial_count,
        "memory_final_entry_count": final_count,
        "evolution_dataset": (
            dataset_info.to_dict()
            if dataset_info is not None
            else evolution_config.dataset.model_dump(mode="json")
        ),
        "mini_batches": batch_records,
        "runs": [
            {
                "sample_id": execution.result.sample_id,
                "status": execution.result.status,
                "outcome": batch_result_outcome(execution.result),
                "turns": execution.result.turns,
                "run_dir": str(execution.result.run_dir),
                "error": execution.result.error,
                "retry_count": execution.retry_count,
                "dataset_index": execution.dataset_index,
                "difficulty": execution.difficulty.value,
                "seed": execution.seed,
            }
            for execution in executions
        ],
    }
    (evolution_dir / "evolution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"Evolution: {evolution_id}",
        f"Memory: {memory_path} | entries {initial_count} -> {final_count}",
        (
            f"Samples: {len(results)} | Completed: {summary['completed']} "
            f"| Failed: {summary['failed']} | Retries: {total_retries}"
        ),
        (
            f"Mini-batch: {evolution_config.run.mini_batch_size} "
            f"| Concurrency: {evolution_config.run.concurrency}"
        ),
    ]
    (evolution_dir / "evolution_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _sample_run_id(sample_id: str, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._") or "sample"
    return f"{slug[:90]}_{index + 1:04d}"


def _discard_run_directory(run_dir: Path, *, evolution_dir: Path) -> None:
    source = run_dir.resolve()
    root = evolution_dir.resolve()
    if source.parent != root:
        raise ValueError(f"refusing to delete run directory outside evolution root: {source}")
    if not source.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {source}")
    shutil.rmtree(source)
