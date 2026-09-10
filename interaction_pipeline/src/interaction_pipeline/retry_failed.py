from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from assistant.config import AssistantConfig
from assistant.config import EnvironmentSettings as AssistantEnvironmentSettings
from assistant.config import load_model_profile as load_assistant_model_profile
from user_simulator.config import EnvironmentSettings as SimulatorEnvironmentSettings
from user_simulator.config import SimulatorConfig
from user_simulator.config import load_model_profile as load_simulator_model_profile
from user_simulator.data.loader import DatasetLoader

from interaction_pipeline.batch import (
    EVALUATION_RETRY_POLICY,
    BatchProgressCallback,
    run_batch,
)
from interaction_pipeline.config import PipelineConfig
from interaction_pipeline.prepare import (
    PreparedPipeline,
    _uniform_simulator_model_profile_path,
)

ALL_FAILURES_FILTER = "all_failures"
INFRASTRUCTURE_FAILURES_ONLY_FILTER = "infrastructure_failures_only"


@dataclass(frozen=True)
class FailedBatchSource:
    directory: Path
    summary: dict[str, Any]
    config: dict[str, Any]
    failed_runs: tuple[dict[str, Any], ...]
    seeds_by_sample_id: dict[str, int]
    failure_filter: str
    source_failed_count: int


def load_failed_batch_source(
    batch_dir: str | Path,
    *,
    infrastructure_failures_only: bool = False,
) -> FailedBatchSource:
    directory = Path(batch_dir).expanduser().resolve()
    summary = _read_json(directory / "batch_summary.json")
    config = _read_yaml(directory / "batch_config.yaml")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise TypeError(f"{directory}/batch_summary.json has no valid runs list")
    all_failed_runs = tuple(
        item for item in runs if isinstance(item, dict) and item.get("status") != "completed"
    )
    if len(all_failed_runs) != int(summary.get("failed", len(all_failed_runs))):
        raise ValueError("batch summary failed count does not match its runs list")
    failure_filter = (
        INFRASTRUCTURE_FAILURES_ONLY_FILTER
        if infrastructure_failures_only
        else ALL_FAILURES_FILTER
    )
    failed_runs = (
        tuple(item for item in all_failed_runs if _is_infrastructure_failure_record(item))
        if infrastructure_failures_only
        else all_failed_runs
    )
    batch_config = config.get("batch")
    if not isinstance(batch_config, dict):
        raise TypeError(f"{directory}/batch_config.yaml has no valid batch mapping")
    if bool(batch_config.get("update_memory")) or bool(summary.get("update_memory")):
        raise ValueError("failed-sample recovery requires a frozen --no-update-memory batch")
    raw_samples = batch_config.get("samples")
    if not isinstance(raw_samples, list):
        raise TypeError("batch config has no valid sample/seed list")
    seeds_by_sample_id: dict[str, int] = {}
    for index, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            raise TypeError(f"batch.samples[{index}] must be a mapping")
        sample_id = item.get("sample_id")
        seed = item.get("seed")
        if not isinstance(sample_id, str) or not isinstance(seed, int):
            raise TypeError(f"batch.samples[{index}] must contain sample_id and integer seed")
        if sample_id in seeds_by_sample_id:
            raise ValueError(f"duplicate sample_id in batch config: {sample_id}")
        seeds_by_sample_id[sample_id] = seed
    missing = [
        item.get("sample_id")
        for item in failed_runs
        if item.get("sample_id") not in seeds_by_sample_id
    ]
    if missing:
        raise ValueError(f"failed samples are absent from batch seed snapshot: {missing}")
    return FailedBatchSource(
        directory=directory,
        summary=summary,
        config=config,
        failed_runs=failed_runs,
        seeds_by_sample_id=seeds_by_sample_id,
        failure_filter=failure_filter,
        source_failed_count=len(all_failed_runs),
    )


def prepare_pipeline_from_batch_snapshot(source: FailedBatchSource) -> PreparedPipeline:
    snapshot = source.config
    try:
        pipeline_config = PipelineConfig.model_validate(snapshot["pipeline"])
        simulator_config = SimulatorConfig.model_validate(snapshot["simulator"]["config"])
        assistant_config = AssistantConfig.model_validate(snapshot["assistant"]["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot reconstruct pipeline from {source.directory}: {exc}") from exc
    _verify_dataset(snapshot, simulator_config.dataset_path)
    _verify_model_profiles(snapshot)
    _verify_memory_store(source)
    action_guard_model_profile_path = None
    action_guard_model_source = None
    guard_snapshot = snapshot.get("assistant", {}).get("action_guard", {})
    if isinstance(guard_snapshot, dict) and guard_snapshot.get("enabled") is True:
        model_snapshot = guard_snapshot.get("model_profile")
        if isinstance(model_snapshot, dict) and isinstance(model_snapshot.get("source_path"), str):
            action_guard_model_profile_path = str(
                Path(model_snapshot["source_path"]).expanduser().resolve()
            )
            action_guard_model_source = str(model_snapshot.get("source") or "batch_snapshot")
        else:
            # Backward compatibility for pre-schema-v2 frozen React batches.
            action_guard_model_profile_path = _uniform_simulator_model_profile_path(
                simulator_config
            )
            action_guard_model_source = "simulator.models.unified_reconstructed"
    return PreparedPipeline(
        config=pipeline_config,
        simulator_config=simulator_config,
        assistant_config=assistant_config,
        dataset=DatasetLoader(simulator_config.dataset_path),
        simulator_environment=SimulatorEnvironmentSettings(),
        assistant_environment=AssistantEnvironmentSettings(),
        action_guard_model_profile_path=action_guard_model_profile_path,
        action_guard_model_source=action_guard_model_source,
    )


async def retry_failed_batch(
    prepared: PreparedPipeline,
    source: FailedBatchSource,
    *,
    output_root: str | Path | None = None,
    concurrency: int | None = None,
    max_additional_attempts: int = 3,
    progress_callback: BatchProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Retry only final failures and emit a non-destructive consolidated batch.

    The source attempt already happened, so ``max_additional_attempts=3`` runs at
    most three new attempts, matching an original ``sample_retries=3`` batch.
    """

    if max_additional_attempts < 1:
        raise ValueError("max_additional_attempts must be at least 1")
    if not source.failed_runs:
        raise ValueError(f"batch has no failed samples: {source.directory}")
    effective_concurrency = concurrency or int(source.summary.get("concurrency", 1))
    if effective_concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    sample_ids = [str(item["sample_id"]) for item in source.failed_runs]
    samples = [prepared.dataset.load_sample_by_id(sample_id) for sample_id in sample_ids]
    seeds = [source.seeds_by_sample_id[sample_id] for sample_id in sample_ids]
    repair_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else source.directory / "repairs"
    )
    # The source run is the initial attempt. Therefore N additional attempts map
    # to one retry-only run plus N-1 retries inside the regular batch executor.
    repair_dir, _ = await run_batch(
        prepared,
        samples,
        output_root=repair_root,
        concurrency=effective_concurrency,
        sample_retries=max_additional_attempts - 1,
        update_memory=False,
        sample_seeds=seeds,
        progress_callback=progress_callback,
    )
    retry_summary_path = repair_dir / "batch_summary.json"
    retry_config_path = repair_dir / "batch_config.yaml"
    retry_summary = _read_json(retry_summary_path)
    retry_config = _read_yaml(retry_config_path)
    shutil.copy2(retry_summary_path, repair_dir / "retry_only_summary.json")
    shutil.copy2(retry_config_path, repair_dir / "retry_only_config.yaml")
    merged = _merge_batch_summaries(
        source,
        retry_summary,
        repair_dir=repair_dir,
        max_additional_attempts=max_additional_attempts,
        concurrency=effective_concurrency,
    )
    retry_summary_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    consolidated_config = _consolidated_config(
        source,
        retry_config,
        merged,
        repair_dir=repair_dir,
        max_additional_attempts=max_additional_attempts,
        concurrency=effective_concurrency,
    )
    with retry_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(consolidated_config, handle, allow_unicode=True, sort_keys=False)
    _write_summary_text(repair_dir / "batch_summary.txt", merged)
    return repair_dir, merged


def _merge_batch_summaries(
    source: FailedBatchSource,
    retry_summary: dict[str, Any],
    *,
    repair_dir: Path,
    max_additional_attempts: int,
    concurrency: int,
) -> dict[str, Any]:
    retry_runs = retry_summary.get("runs")
    if not isinstance(retry_runs, list):
        raise TypeError("retry-only batch summary has no valid runs list")
    if not all(isinstance(item, dict) for item in retry_runs):
        raise TypeError("retry-only batch runs must all be mappings")
    replacements = {str(item["sample_id"]): item for item in retry_runs}
    expected = {str(item["sample_id"]) for item in source.failed_runs}
    if set(replacements) != expected:
        raise ValueError("retry-only results do not match the source failed-sample set")
    merged_runs: list[dict[str, Any]] = []
    recovery_attempts = 0
    recovered = 0
    selected_still_failed = 0
    for original in source.summary["runs"]:
        sample_id = str(original["sample_id"])
        replacement = replacements.get(sample_id)
        if replacement is None:
            merged_runs.append(deepcopy(original))
            continue
        attempts_used = int(replacement.get("retry_count", 0)) + 1
        recovery_attempts += attempts_used
        updated = deepcopy(replacement)
        updated["retry_count"] = int(original.get("retry_count", 0)) + attempts_used
        updated["recovery"] = {
            "source_run_dir": original.get("run_dir"),
            "source_status": original.get("status"),
            "source_outcome": original.get("outcome"),
            "source_error": original.get("error"),
            "attempts_used": attempts_used,
        }
        if updated.get("status") == "completed":
            recovered += 1
        else:
            selected_still_failed += 1
        merged_runs.append(updated)
    failed = sum(item.get("status") != "completed" for item in merged_runs)
    behavioral = sum(
        item.get("status") != "completed"
        and isinstance(item.get("error"), str)
        and item["error"].startswith("EpisodeTurnLimitError:")
        for item in merged_runs
    )
    source_total_attempts = int(
        source.summary.get(
            "total_attempts",
            int(source.summary.get("sample_count", len(merged_runs)))
            + int(source.summary.get("total_retries", 0)),
        )
    )
    source_total_retries = int(source.summary.get("total_retries", 0))
    now = datetime.now(UTC).isoformat()
    return {
        "batch_id": retry_summary.get("batch_id", repair_dir.name),
        "created_at": retry_summary.get("created_at", now),
        "completed_at": now,
        "concurrency": concurrency,
        "sample_retries": max_additional_attempts,
        "retry_policy": EVALUATION_RETRY_POLICY,
        "update_memory": False,
        "sample_count": len(merged_runs),
        "total_attempts": source_total_attempts + recovery_attempts,
        "total_retries": source_total_retries + recovery_attempts,
        "retried_samples": sum(int(item.get("retry_count", 0)) > 0 for item in merged_runs),
        "completed": len(merged_runs) - failed,
        "behavioral_failures": behavioral,
        "infrastructure_failures": failed - behavioral,
        "failed": failed,
        "recovery": {
            "source_batch_id": source.summary.get("batch_id"),
            "source_batch_dir": str(source.directory),
            "failure_filter": source.failure_filter,
            "source_failed_samples": source.source_failed_count,
            "selected_failed_samples": len(source.failed_runs),
            "unselected_failed_samples": source.source_failed_count - len(source.failed_runs),
            "max_additional_attempts": max_additional_attempts,
            "attempts_used": recovery_attempts,
            "recovered": recovered,
            "still_failed": selected_still_failed,
            "remaining_batch_failures": failed,
            "retry_only_summary": str(repair_dir / "retry_only_summary.json"),
        },
        "runs": merged_runs,
    }


def _consolidated_config(
    source: FailedBatchSource,
    retry_config: dict[str, Any],
    summary: dict[str, Any],
    *,
    repair_dir: Path,
    max_additional_attempts: int,
    concurrency: int,
) -> dict[str, Any]:
    config = deepcopy(source.config)
    batch = config.setdefault("batch", {})
    batch.update(
        {
            "batch_id": summary["batch_id"],
            "created_at": summary["created_at"],
            "concurrency": concurrency,
            "sample_retries": max_additional_attempts,
            "retry_policy": EVALUATION_RETRY_POLICY,
            "update_memory": False,
            "sample_count": summary["sample_count"],
            "recovery": summary["recovery"],
        }
    )
    pipeline_run = config.get("pipeline", {}).get("run")
    if isinstance(pipeline_run, dict):
        pipeline_run["concurrency"] = concurrency
        pipeline_run["sample_retries"] = max_additional_attempts
        pipeline_run["update_memory"] = False
        pipeline_run["output_dir"] = str(repair_dir)
    config["recovery_execution_snapshot"] = retry_config
    return config


def _write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    recovery = summary["recovery"]
    lines = [
        f"Recovered batch: {summary['batch_id']}",
        f"Source batch: {recovery['source_batch_id']} | {recovery['source_batch_dir']}",
        (
            f"Samples: {summary['sample_count']} | Completed: {summary['completed']} "
            f"| Turn-limit failures: {summary['behavioral_failures']} "
            f"| Infrastructure failures: {summary['infrastructure_failures']}"
        ),
        (
            f"Failure filter: {recovery['failure_filter']} "
            f"| Failed samples selected: {recovery['selected_failed_samples']} "
            f"| Recovery attempts used: {recovery['attempts_used']} "
            f"| Recovered: {recovery['recovered']} "
            f"| Selected still failed: {recovery['still_failed']} "
            f"| Remaining batch failures: {recovery['remaining_batch_failures']}"
        ),
        "",
    ]
    for item in summary["runs"]:
        marker = "RERUN" if "recovery" in item else "REUSED"
        lines.append(
            f"[{marker}/{item['status'].upper()}] {item['sample_id']} "
            f"| retries={item.get('retry_count', 0)} | turns={item.get('turns')} "
            f"| {item.get('run_dir')}" + (f" | {item['error']}" if item.get("error") else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_infrastructure_failure_record(item: dict[str, Any]) -> bool:
    """Classify a final batch record, including snapshots predating ``outcome``."""

    outcome = item.get("outcome")
    if outcome is not None:
        return outcome == "INFRASTRUCTURE_FAILURE"
    error = item.get("error")
    return not (
        isinstance(error, str) and error.startswith("EpisodeTurnLimitError:")
    )


def _verify_dataset(snapshot: dict[str, Any], dataset_path: str) -> None:
    dataset = snapshot.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("sha256"), str):
        raise TypeError("batch snapshot has no dataset SHA-256 provenance")
    path = Path(dataset_path).expanduser().resolve()
    actual = _sha256_file(path)
    if actual != dataset["sha256"]:
        raise ValueError(
            f"dataset changed since the source batch: expected {dataset['sha256']}, got {actual}"
        )


def _verify_model_profiles(snapshot: dict[str, Any]) -> None:
    simulator_profiles = snapshot.get("simulator", {}).get("model_profiles")
    if not isinstance(simulator_profiles, dict):
        raise TypeError("batch snapshot has no simulator model-profile provenance")
    for component, item in simulator_profiles.items():
        if not isinstance(item, dict):
            raise TypeError(f"invalid simulator model-profile snapshot for {component}")
        current = load_simulator_model_profile(item["source_path"]).model_dump(mode="json")
        if current != item.get("settings"):
            raise ValueError(f"simulator model profile changed since source batch: {component}")
    assistant_profile = snapshot.get("assistant", {}).get("model_profile")
    if not isinstance(assistant_profile, dict):
        raise TypeError("batch snapshot has no assistant model-profile provenance")
    current = load_assistant_model_profile(assistant_profile["source_path"]).model_dump(mode="json")
    if current != assistant_profile.get("settings"):
        raise ValueError("assistant model profile changed since source batch")


def _verify_memory_store(source: FailedBatchSource) -> None:
    assistant = source.config.get("assistant")
    if not isinstance(assistant, dict):
        raise TypeError("batch snapshot has no assistant provenance")
    profile_snapshot = assistant.get("model_profile")
    if not isinstance(profile_snapshot, dict):
        raise TypeError("batch snapshot has no assistant model-profile provenance")
    profile = load_assistant_model_profile(profile_snapshot["source_path"])
    if profile.memory is None:
        return
    memory_path = profile.memory.path
    if not memory_path.is_file():
        raise ValueError(f"assistant memory store is missing: {memory_path}")
    provenance = assistant.get("memory_store")
    if isinstance(provenance, dict) and isinstance(provenance.get("sha256"), str):
        if Path(provenance.get("path", "")).expanduser().resolve() != memory_path.resolve():
            raise ValueError("assistant memory path changed since source batch")
        actual = _sha256_file(memory_path)
        if actual != provenance["sha256"]:
            raise ValueError(
                "assistant memory changed since source batch: "
                f"expected {provenance['sha256']}, got {actual}"
            )
        return
    # Older snapshots did not include a memory hash. They can still be safely
    # recovered when filesystem metadata proves the file predates the batch.
    created_at = source.summary.get("created_at")
    if not isinstance(created_at, str):
        raise TypeError("cannot verify memory stability for this legacy batch")
    batch_started_at = datetime.fromisoformat(created_at)
    memory_modified_at = datetime.fromtimestamp(memory_path.stat().st_mtime, tz=UTC)
    if memory_modified_at > batch_started_at:
        raise ValueError(
            "legacy batch has no memory SHA-256 and the memory file was modified after "
            f"the batch started ({memory_modified_at.isoformat()} > {batch_started_at.isoformat()})"
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required batch artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required batch artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value
