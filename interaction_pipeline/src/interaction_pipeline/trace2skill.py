from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml
from assistant.config import ModelProfile, load_model_profile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from user_simulator.audit.logger import read_git_commit
from user_simulator.domain.dag import Sample

from interaction_pipeline.core import RunResult, record_setup_failure, run_interaction
from interaction_pipeline.evolution_dataset import (
    EvolutionDatasetInfo,
    sample_evolution_difficulty,
)
from interaction_pipeline.prepare import PreparedPipeline
from interaction_pipeline.trace2skill_config import Trace2SkillConfig, Trace2SkillRunSettings

try:
    import fcntl
except ImportError:  # pragma: no cover - benchmark runs use Linux
    fcntl = None  # type: ignore[assignment]

SUCCESS = "SUCCESS"
FAILURE_TURN_LIMIT = "FAILURE_TURN_LIMIT"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
NO_SKILL = "No Skill"
SCHEMA_VERSION = 1
COLLECTION_ARTIFACT_TYPE = "trace2skill_trajectory_collection"
COLLECTION_SLOT_ARTIFACT_TYPE = "trace2skill_visible_trajectory"
BUILD_ARTIFACT_TYPE = "trace2skill_skill_build"

TrajectoryOutcome = Literal["SUCCESS", "FAILURE_TURN_LIMIT"]
RunClassification = Literal[
    "SUCCESS",
    "FAILURE_TURN_LIMIT",
    "INFRASTRUCTURE_FAILURE",
]
TraceProgressCallback = Callable[[int, int, dict[str, Any]], None]
Trace2SkillRunMode = Literal["canonical", "smoke"]
T = TypeVar("T")


class VisibleMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class VisibleTrajectory(BaseModel):
    """The complete and only model-visible per-trajectory payload."""

    model_config = ConfigDict(extra="forbid")

    outcome: TrajectoryOutcome
    dialogue: tuple[VisibleMessage, ...]

    @model_validator(mode="after")
    def validate_dialogue(self) -> VisibleTrajectory:
        if not self.dialogue or len(self.dialogue) % 2:
            raise ValueError("visible dialogue must contain complete user/assistant pairs")
        for index, message in enumerate(self.dialogue):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("visible dialogue must alternate user then assistant")
        return self

    @property
    def assistant_turns(self) -> int:
        return len(self.dialogue) // 2


@dataclass(frozen=True)
class Trace2SkillEvolutionResult:
    run_dir: Path
    complete: bool
    valid_trajectory_count: int
    patch_count: int
    final_skill_path: Path | None = None
    published_skill_path: Path | None = None
    run_mode: Trace2SkillRunMode = "canonical"
    canonical: bool = True
    skipped_count: int = 0
    stage_error: str | None = None


@dataclass(frozen=True)
class Trace2SkillCollectionResult:
    run_dir: Path
    complete: bool
    valid_trajectory_count: int
    pending_count: int
    exhausted_pending_count: int = 0
    recovery_grant_count: int = 0
    corpus_sha256: str | None = None
    run_mode: Trace2SkillRunMode = "canonical"
    canonical: bool = True
    skill_build_allowed: bool = True


@dataclass(frozen=True)
class Trace2SkillCorpus:
    collection_dir: Path
    manifest: dict[str, Any]
    seal: dict[str, Any]
    slots: tuple[dict[str, Any], ...]
    trajectories: tuple[VisibleTrajectory, ...]
    trajectory_sha256: tuple[str, ...]

    @property
    def corpus_sha256(self) -> str:
        return cast(str, self.seal["corpus_sha256"])


@dataclass(frozen=True)
class Trace2SkillRunContract:
    run_mode: Trace2SkillRunMode
    canonical: bool
    evaluation_allowed: bool
    selected_sample_count: int
    selection: Literal["complete_source_order", "source_order_prefix"]
    limit: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "canonical": self.canonical,
            "evaluation_allowed": self.evaluation_allowed,
            "selected_sample_count": self.selected_sample_count,
            "selection": self.selection,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class Trace2SkillCollectionContract:
    run_mode: Trace2SkillRunMode
    canonical: bool
    skill_build_allowed: bool
    selected_sample_count: int
    source_dataset_sample_count: int
    selection: Literal["complete_source_order", "source_order_prefix"]
    limit: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "canonical": self.canonical,
            "skill_build_allowed": self.skill_build_allowed,
            "selected_sample_count": self.selected_sample_count,
            "source_dataset_sample_count": self.source_dataset_sample_count,
            "selection": self.selection,
            "limit": self.limit,
        }


class Trace2SkillError(RuntimeError):
    pass


class SkillPublicationError(Trace2SkillError):
    pass


def classify_run_result(result: RunResult, *, turn_budget: int = 20) -> RunClassification:
    if result.status == "completed" and result.error is None:
        if 1 <= result.turns <= turn_budget:
            return SUCCESS
        return INFRASTRUCTURE_FAILURE
    if (
        result.status != "completed"
        and result.turns == turn_budget
        and isinstance(result.error, str)
        and result.error.startswith("EpisodeTurnLimitError:")
    ):
        return FAILURE_TURN_LIMIT
    return INFRASTRUCTURE_FAILURE


def canonical_visible_trajectory(
    transcript_path: str | Path,
    *,
    outcome: TrajectoryOutcome,
    expected_turns: int,
    turn_budget: int = 20,
) -> VisibleTrajectory:
    """Read accepted pairs only, dropping the unmatched post-budget user message."""

    path = Path(transcript_path)
    rows: list[VisibleMessage] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"line {line_number} is not an object")
                role = value.get("role")
                content = value.get("content")
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    raise ValueError(f"line {line_number} is not a visible dialogue message")
                rows.append(VisibleMessage(role=role, content=content))
    except (OSError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"cannot read canonical visible transcript: {exc}") from exc

    paired_length = len(rows)
    if paired_length % 2:
        if rows[-1].role != "user":
            raise ValueError("only a trailing unmatched user message may be discarded")
        paired_length -= 1
    paired = rows[:paired_length]
    if len(paired) // 2 != expected_turns:
        raise ValueError(
            f"accepted transcript has {len(paired) // 2} turns; expected {expected_turns}"
        )
    if expected_turns > turn_budget:
        raise ValueError("accepted transcript exceeds the turn budget")
    if outcome == FAILURE_TURN_LIMIT and expected_turns != turn_budget:
        raise ValueError("turn-limit failure must contain exactly the full turn budget")
    if outcome == SUCCESS and len(rows) != paired_length:
        raise ValueError("successful transcript must not contain a trailing user message")
    return VisibleTrajectory(outcome=outcome, dialogue=tuple(paired))


def fixed_chunks(items: Sequence[T], size: int = 32) -> list[list[T]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


async def run_trace2skill_collection(
    prepared: PreparedPipeline,
    samples: list[Sample],
    *,
    output_root: str | Path,
    trace_config: Trace2SkillConfig,
    dataset_info: EvolutionDatasetInfo,
    resume_dir: str | Path | None = None,
    progress_callback: TraceProgressCallback | None = None,
    limit: int | None = None,
    retry_exhausted: bool = False,
) -> Trace2SkillCollectionResult:
    """Collect and seal a model-specific visible-trajectory corpus.

    This stage never constructs a distillation client and never runs analyst,
    merge, APPLY, publication, or evaluation work. A limited source-order prefix
    is an audit-only collection and cannot be consumed by the Skill build stage.
    """

    if retry_exhausted and resume_dir is None:
        raise ValueError("retry_exhausted requires an existing collection resume_dir")

    settings = trace_config.run
    _validate_trace_run(prepared, samples, trace_config, dataset_info)
    collection_contract = _resolve_collection_contract(
        expected_sample_count=trace_config.dataset.expected_sample_count,
        limit=limit,
    )
    selected_samples = samples[: collection_contract.selected_sample_count]
    profile_path = prepared.assistant_config.models["assistant"]
    run_dir, created_at = _prepare_collection_directory(
        output_root=Path(output_root),
        resume_dir=Path(resume_dir) if resume_dir is not None else None,
        prepared=prepared,
        trace_config=trace_config,
        dataset_info=dataset_info,
        samples=selected_samples,
        profile_path=profile_path,
        collection_contract=collection_contract,
    )
    slots_dir = run_dir / "trajectory_slots"
    slots_dir.mkdir(exist_ok=True)
    difficulties = [
        sample_evolution_difficulty(sample, field=trace_config.dataset.difficulty_field)
        for sample in selected_samples
    ]
    semaphore = asyncio.Semaphore(settings.trajectory_concurrency)

    async def collect(index: int, sample: Sample) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            slot = await _collect_trajectory_slot(
                prepared=prepared,
                sample=sample,
                dataset_index=index,
                difficulty=difficulties[index].value,
                run_dir=run_dir,
                slot_path=slots_dir / f"{index + 1:04d}.json",
                settings=settings,
                dataset_sha256=dataset_info.sha256,
                retry_exhausted=retry_exhausted,
            )
            return index, slot

    tasks = [
        asyncio.create_task(collect(index, sample)) for index, sample in enumerate(selected_samples)
    ]
    ordered_slots: list[dict[str, Any] | None] = [None] * len(selected_samples)
    try:
        for finished, task in enumerate(asyncio.as_completed(tasks), 1):
            index, slot = await task
            ordered_slots[index] = slot
            if progress_callback is not None:
                progress_callback(finished, len(selected_samples), slot)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    slots = [cast(dict[str, Any], item) for item in ordered_slots]
    pending = [item for item in slots if item.get("status") != "complete"]
    valid_count = len(slots) - len(pending)
    exhausted_pending_count = sum(
        item.get("exhausted_infrastructure_attempts") is True for item in pending
    )
    recovery_grant_count = sum(
        len(item.get("infrastructure_recovery_grants", [])) for item in slots
    )
    if pending:
        _write_collection_summary(
            run_dir,
            created_at=created_at,
            complete=False,
            slots=slots,
            collection_contract=collection_contract,
        )
        return Trace2SkillCollectionResult(
            run_dir=run_dir,
            complete=False,
            valid_trajectory_count=valid_count,
            pending_count=len(pending),
            exhausted_pending_count=exhausted_pending_count,
            recovery_grant_count=recovery_grant_count,
            run_mode=collection_contract.run_mode,
            canonical=collection_contract.canonical,
            skill_build_allowed=collection_contract.skill_build_allowed,
        )

    seal = _seal_trajectory_corpus(
        run_dir,
        slots=slots,
        manifest=_read_json(run_dir / "manifest.json"),
    )
    _write_collection_summary(
        run_dir,
        created_at=created_at,
        complete=True,
        slots=slots,
        collection_contract=collection_contract,
        corpus_sha256=cast(str, seal["corpus_sha256"]),
    )
    return Trace2SkillCollectionResult(
        run_dir=run_dir,
        complete=True,
        valid_trajectory_count=valid_count,
        pending_count=0,
        exhausted_pending_count=0,
        recovery_grant_count=recovery_grant_count,
        corpus_sha256=cast(str, seal["corpus_sha256"]),
        run_mode=collection_contract.run_mode,
        canonical=collection_contract.canonical,
        skill_build_allowed=collection_contract.skill_build_allowed,
    )


def load_trace2skill_corpus(
    collection_dir: str | Path,
    *,
    require_build_eligible: bool = True,
) -> Trace2SkillCorpus:
    """Load a sealed corpus and verify every whitelisted trajectory payload."""

    source = Path(collection_dir).expanduser().resolve()
    manifest = _read_json(source / "manifest.json")
    if manifest.get("artifact_type") != COLLECTION_ARTIFACT_TYPE:
        raise Trace2SkillError(f"not a Trace2Skill trajectory collection: {source}")
    seal = _read_json(source / "corpus.json")
    if seal.get("artifact_type") != COLLECTION_ARTIFACT_TYPE or seal.get("status") != "sealed":
        raise Trace2SkillError(f"trajectory corpus is not sealed: {source}")
    if seal.get("collection_fingerprint") != manifest.get("immutable_fingerprint"):
        raise Trace2SkillError("trajectory corpus seal does not match its collection manifest")

    collection_contract = _normalize_collection_contract(manifest)
    expected_count = collection_contract.selected_sample_count
    if require_build_eligible and not collection_contract.skill_build_allowed:
        raise Trace2SkillError(
            "trial trajectory collection is audit-only and cannot be used for Skill build; "
            "run collect without --limit to create the canonical corpus"
        )
    sample_ids = manifest.get("sample_ids")
    dataset = manifest.get("development_dataset")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != expected_count
        or not all(isinstance(item, str) and item for item in sample_ids)
    ):
        raise Trace2SkillError("trajectory collection manifest has invalid sample order")
    if len(set(sample_ids)) != len(sample_ids):
        raise Trace2SkillError("trajectory collection manifest contains duplicate sample IDs")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("sha256"), str):
        raise Trace2SkillError("trajectory collection manifest has no dataset hash")
    expected_order_hash = _hash_text("\n".join(sample_ids))
    if manifest.get("sample_order_sha256") != expected_order_hash:
        raise Trace2SkillError("trajectory collection sample order hash is invalid")

    slots_dir = source / "trajectory_slots"
    paths = sorted(slots_dir.glob("*.json"))
    expected_names = [f"{index + 1:04d}.json" for index in range(expected_count)]
    if [path.name for path in paths] != expected_names:
        raise Trace2SkillError(
            f"trajectory collection does not have exactly {expected_count} ordered slots"
        )
    slots: list[dict[str, Any]] = []
    trajectories: list[VisibleTrajectory] = []
    entry_hashes: list[str] = []
    for index, (path, sample_id) in enumerate(zip(paths, sample_ids, strict=True)):
        slot = _read_json(path)
        if (
            slot.get("schema_version") != SCHEMA_VERSION
            or slot.get("artifact_type") != COLLECTION_SLOT_ARTIFACT_TYPE
            or slot.get("status") != "complete"
            or slot.get("dataset_index") != index
            or slot.get("sample_id") != sample_id
            or slot.get("dataset_sha256") != dataset["sha256"]
        ):
            raise Trace2SkillError(f"invalid sealed trajectory slot: {path}")
        trajectory = VisibleTrajectory.model_validate(slot.get("visible_trajectory"))
        if (
            slot.get("outcome") != trajectory.outcome
            or slot.get("turns") != trajectory.assistant_turns
        ):
            raise Trace2SkillError(f"trajectory slot outcome/turn count mismatch: {path}")
        entry_hash = _trajectory_entry_hash(
            dataset_index=index,
            sample_id=sample_id,
            dataset_sha256=cast(str, dataset["sha256"]),
            trajectory=trajectory,
        )
        if slot.get("trajectory_sha256") != entry_hash:
            raise Trace2SkillError(f"trajectory slot hash mismatch: {path}")
        slots.append(slot)
        trajectories.append(trajectory)
        entry_hashes.append(entry_hash)

    corpus_sha256 = _hash_text(_canonical_json(entry_hashes))
    if (
        seal.get("sample_count") != expected_count
        or seal.get("dataset_sha256") != dataset["sha256"]
        or seal.get("sample_order_sha256") != expected_order_hash
        or seal.get("trajectory_sha256") != entry_hashes
        or seal.get("corpus_sha256") != corpus_sha256
    ):
        raise Trace2SkillError("trajectory corpus seal failed content verification")
    for key, expected in {
        "run_mode": collection_contract.run_mode,
        "canonical": collection_contract.canonical,
        "skill_build_allowed": collection_contract.skill_build_allowed,
        "source_dataset_sample_count": collection_contract.source_dataset_sample_count,
    }.items():
        if key in seal and seal.get(key) != expected:
            raise Trace2SkillError("trajectory corpus seal has an invalid collection contract")
    return Trace2SkillCorpus(
        collection_dir=source,
        manifest=manifest,
        seal=seal,
        slots=tuple(slots),
        trajectories=tuple(trajectories),
        trajectory_sha256=tuple(entry_hashes),
    )


async def _collect_trajectory_slot(
    *,
    prepared: PreparedPipeline,
    sample: Sample,
    dataset_index: int,
    difficulty: str,
    run_dir: Path,
    slot_path: Path,
    settings: Trace2SkillRunSettings,
    dataset_sha256: str,
    retry_exhausted: bool = False,
) -> dict[str, Any]:
    existing = _read_slot(slot_path, dataset_index=dataset_index, sample_id=sample.sample_id)
    if existing is not None:
        if existing.get("artifact_type") != COLLECTION_SLOT_ARTIFACT_TYPE:
            raise Trace2SkillError(f"incompatible collection slot artifact: {slot_path}")
        if existing.get("status") == "complete":
            trajectory = VisibleTrajectory.model_validate(existing.get("visible_trajectory"))
            expected_hash = _trajectory_entry_hash(
                dataset_index=dataset_index,
                sample_id=sample.sample_id,
                dataset_sha256=dataset_sha256,
                trajectory=trajectory,
            )
            if existing.get("trajectory_sha256") != expected_hash:
                raise Trace2SkillError(f"completed collection slot hash mismatch: {slot_path}")
            return existing

    slot = existing or {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COLLECTION_SLOT_ARTIFACT_TYPE,
        "dataset_index": dataset_index,
        "sample_id": sample.sample_id,
        "difficulty": difficulty,
        "seed": settings.seed + dataset_index,
        "dataset_sha256": dataset_sha256,
        "status": "pending_trajectory",
        "episode_attempts": [],
        "infrastructure_attempt_limit": settings.infrastructure_max_attempts,
    }
    trajectory: VisibleTrajectory | None = None
    if slot.get("visible_trajectory") is not None:
        trajectory = VisibleTrajectory.model_validate(slot["visible_trajectory"])

    if trajectory is None:
        attempts_value = slot.get("episode_attempts")
        if not isinstance(attempts_value, list):
            raise Trace2SkillError(f"collection slot has invalid attempt history: {slot_path}")
        attempts = cast(list[dict[str, Any]], attempts_value)
        attempt_limit_value = slot.get("infrastructure_attempt_limit")
        if attempt_limit_value is None:
            # Collections created before explicit recovery recorded only the fixed base limit.
            attempt_limit = max(settings.infrastructure_max_attempts, len(attempts))
        elif (
            not isinstance(attempt_limit_value, int)
            or isinstance(attempt_limit_value, bool)
            or attempt_limit_value < settings.infrastructure_max_attempts
            or attempt_limit_value < len(attempts)
        ):
            raise Trace2SkillError(f"collection slot has invalid attempt limit: {slot_path}")
        else:
            attempt_limit = attempt_limit_value
        slot["infrastructure_attempt_limit"] = attempt_limit

        if retry_exhausted and existing is not None and len(attempts) >= attempt_limit:
            grants_value = slot.get("infrastructure_recovery_grants", [])
            if not isinstance(grants_value, list):
                raise Trace2SkillError(f"collection slot has invalid recovery history: {slot_path}")
            grants = cast(list[dict[str, Any]], grants_value)
            additional_attempts = settings.infrastructure_max_attempts
            new_attempt_limit = attempt_limit + additional_attempts
            grants.append(
                {
                    "grant_number": len(grants) + 1,
                    "granted_at": datetime.now(UTC).isoformat(),
                    "reason": "explicit_retry_exhausted_resume",
                    "attempts_already_used": len(attempts),
                    "previous_attempt_limit": attempt_limit,
                    "additional_attempts": additional_attempts,
                    "new_attempt_limit": new_attempt_limit,
                }
            )
            slot["infrastructure_recovery_grants"] = grants
            slot["infrastructure_attempt_limit"] = new_attempt_limit
            slot.pop("exhausted_infrastructure_attempts", None)
            attempt_limit = new_attempt_limit
            _atomic_json(slot_path, slot)

        for attempt_number in range(
            len(attempts) + 1,
            attempt_limit + 1,
        ):
            run_id = f"trajectory_{dataset_index + 1:04d}_attempt_{attempt_number:02d}"
            attempt_record = {
                "attempt_number": attempt_number,
                "classification": INFRASTRUCTURE_FAILURE,
                "status": "running",
                "turns": 0,
                "run_dir": str(run_dir / run_id),
                "error": None,
                "selected_for_corpus": False,
            }
            attempts.append(attempt_record)
            slot["status"] = "pending_trajectory"
            slot.pop("exhausted_infrastructure_attempts", None)
            _atomic_json(slot_path, slot)
            try:
                built = prepared.build(
                    sample,
                    output_dir=run_dir,
                    seed=settings.seed + dataset_index,
                    run_id=run_id,
                    difficulty=difficulty,
                    execution_context={
                        "mode": "trace2skill_visible_parallel_collection",
                        "dataset_index": dataset_index,
                        "attempt_number": attempt_number,
                        "initial_skill": NO_SKILL,
                        "memory_updates": False,
                        "distillation": False,
                    },
                )
                result = await run_interaction(
                    episode=built.episode,
                    assistant=built.assistant,
                    audit=built.audit,
                    update_memory=False,
                    stop_before_over_budget_generation=True,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one development slot
                result = _record_uncaught_attempt(
                    sample_id=sample.sample_id,
                    run_dir=run_dir,
                    run_id=run_id,
                    error=exc,
                )
            classification = classify_run_result(result, turn_budget=settings.turn_budget)
            attempt_record.update(
                {
                    "classification": classification,
                    "status": result.status,
                    "turns": result.turns,
                    "run_dir": str(result.run_dir),
                    "error": result.error,
                }
            )
            if classification == INFRASTRUCTURE_FAILURE:
                slot["last_error"] = result.error or "unscorable infrastructure failure"
                _atomic_json(slot_path, slot)
                continue
            try:
                trajectory = canonical_visible_trajectory(
                    result.run_dir / "transcript.jsonl",
                    outcome=cast(TrajectoryOutcome, classification),
                    expected_turns=result.turns,
                    turn_budget=settings.turn_budget,
                )
            except ValueError as exc:
                attempt_record["classification"] = INFRASTRUCTURE_FAILURE
                attempt_record["canonicalization_error"] = str(exc)
                slot["last_error"] = "canonical visible transcript validation failed"
                _atomic_json(slot_path, slot)
                trajectory = None
                continue
            trajectory_hash = _trajectory_entry_hash(
                dataset_index=dataset_index,
                sample_id=sample.sample_id,
                dataset_sha256=dataset_sha256,
                trajectory=trajectory,
            )
            slot.update(
                {
                    "status": "complete",
                    "outcome": classification,
                    "turns": result.turns,
                    "selected_run_dir": str(result.run_dir),
                    "selected_attempt_number": attempt_number,
                    "retry_count": attempt_number - 1,
                    "visible_trajectory": trajectory.model_dump(mode="json"),
                    "trajectory_sha256": trajectory_hash,
                }
            )
            attempt_record["selected_for_corpus"] = True
            slot.pop("last_error", None)
            slot.pop("exhausted_infrastructure_attempts", None)
            _atomic_json(slot_path, slot)
            return slot

    if trajectory is not None:
        trajectory_hash = _trajectory_entry_hash(
            dataset_index=dataset_index,
            sample_id=sample.sample_id,
            dataset_sha256=dataset_sha256,
            trajectory=trajectory,
        )
        slot.update(
            {
                "status": "complete",
                "outcome": trajectory.outcome,
                "turns": trajectory.assistant_turns,
                "trajectory_sha256": trajectory_hash,
            }
        )
        _atomic_json(slot_path, slot)
        return slot

    slot["status"] = "pending_trajectory"
    slot["exhausted_infrastructure_attempts"] = True
    _atomic_json(slot_path, slot)
    return slot


def _write_assistant_override(run_dir: Path, *, skill_path: Path, profile_path: str) -> None:
    override = {
        # Keep this run-specific fallback pinned to its immutable run artifact,
        # even when the model profile also points at a newer published skill.
        "components": {"baseline": "trace2skill", "profile_skill_enabled": False},
        "prompts": {"trace2skill": str(skill_path.resolve())},
        "models": {"assistant": str(Path(profile_path).resolve())},
    }
    _atomic_text(
        run_dir / "assistant_trace2skill.yaml",
        yaml.safe_dump(override, allow_unicode=True, sort_keys=False),
    )


def _publish_profile_bound_skill(
    *,
    run_dir: Path,
    run_skill_path: Path,
    profile: ModelProfile,
    profile_path: str,
    dataset_info: EvolutionDatasetInfo | None,
    run_contract: Trace2SkillRunContract,
    development_dataset_sha256: str | None = None,
    source_corpus: Trace2SkillCorpus | None = None,
) -> Path | None:
    """Atomically publish only a canonical skill to its model-specific path."""

    if not run_contract.canonical or profile.skill is None:
        return None
    if profile.skill.framework != "trace2skill":
        raise SkillPublicationError(
            f"unsupported profile-bound skill framework: {profile.skill.framework!r}"
        )
    target = profile.skill.path.expanduser().resolve()
    provenance_path = target.with_name("SKILL.provenance.json")
    lock_path = target.with_name("SKILL.publish.lock")
    run_publication_path = run_dir / "published_skill.json"
    try:
        dataset_sha256 = (
            dataset_info.sha256 if dataset_info is not None else development_dataset_sha256
        )
        if not isinstance(dataset_sha256, str) or not dataset_sha256:
            raise SkillPublicationError("publication requires a development dataset hash")
        content = run_skill_path.read_text(encoding="utf-8")
        skill_sha256 = _hash_text(content)
        profile_source = Path(profile_path).expanduser().resolve()
        manifest = _read_json(run_dir / "manifest.json")
        manifest_profile = manifest.get("assistant_model_profile")
        expected_profile_sha256 = (
            manifest_profile.get("source_sha256") if isinstance(manifest_profile, dict) else None
        )
        current_profile_sha256 = _sha256_file(profile_source)
        if expected_profile_sha256 != current_profile_sha256:
            raise SkillPublicationError("assistant model profile changed during Skill build")
        record_basis = {
            "schema_version": SCHEMA_VERSION,
            "method": "Trace2Skill-Visible-Combined-Parallel-B32",
            "canonical": True,
            "evaluation_allowed": True,
            "profile_name": profile.profile_name,
            "model_id": profile.model_id,
            "model_profile_path": str(profile_source),
            "model_profile_sha256": current_profile_sha256,
            "skill_path": str(target),
            "skill_sha256": skill_sha256,
            "source_run_id": manifest.get("run_id"),
            "source_run_dir": str(run_dir.resolve()),
            "source_skill_path": str(run_skill_path.resolve()),
            "development_dataset_sha256": dataset_sha256,
            "development_sample_count": run_contract.selected_sample_count,
            "source_trajectory_collection_dir": (
                str(source_corpus.collection_dir) if source_corpus is not None else None
            ),
            "source_trajectory_collection_id": (
                source_corpus.manifest.get("run_id") if source_corpus is not None else None
            ),
            "source_trajectory_corpus_sha256": (
                source_corpus.corpus_sha256 if source_corpus is not None else None
            ),
        }
        with _locked_publication(lock_path):
            target.parent.mkdir(parents=True, exist_ok=True)
            if run_publication_path.exists():
                prior_run_record = _read_json(run_publication_path)
                target_record = _read_json(provenance_path) if provenance_path.exists() else {}
                reusable = (
                    prior_run_record.get("source_run_id") == manifest.get("run_id")
                    and prior_run_record.get("skill_path") == str(target)
                    and prior_run_record.get("skill_sha256") == skill_sha256
                    and prior_run_record.get("model_profile_sha256") == current_profile_sha256
                    and target_record.get("source_run_id") == manifest.get("run_id")
                    and target_record.get("skill_sha256") == skill_sha256
                    and target.exists()
                    and _sha256_file(target) == skill_sha256
                )
                if reusable:
                    return target
                raise SkillPublicationError(
                    "this build was already published but the profile-bound skill or its "
                    "provenance has since changed; refusing to roll it back during resume"
                )
            record = {
                **record_basis,
                "previous_skill_sha256": _sha256_file(target) if target.exists() else None,
                "published_at": datetime.now(UTC).isoformat(),
            }
            # Write provenance first and the skill last. A reader can therefore
            # reject a transient hash mismatch, while the final visible commit is
            # the atomic replacement of SKILL.md itself.
            _atomic_json(provenance_path, record)
            _atomic_text(target, content)
            if _sha256_file(target) != skill_sha256:
                raise SkillPublicationError("published skill hash verification failed")
            _atomic_json(run_publication_path, record)
        return target
    except SkillPublicationError:
        raise
    except (OSError, Trace2SkillError) as exc:
        raise SkillPublicationError(
            f"cannot publish profile-bound skill to {target}: {exc}"
        ) from exc


@contextmanager
def _locked_publication(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _profile_behavior_snapshot(profile: ModelProfile) -> dict[str, Any]:
    return {
        "profile_name": profile.profile_name,
        "provider": profile.provider,
        "model_id": profile.model_id,
        "base_url": profile.base_url,
        "routing": profile.routing,
        "reasoning": profile.reasoning.model_dump(mode="json"),
        "generation": profile.generation["assistant"].model_dump(mode="json"),
        "retry": profile.retry.model_dump(mode="json"),
    }


def _profile_behavior_sha256(profile: ModelProfile) -> str:
    return _hash_text(_canonical_json(_profile_behavior_snapshot(profile)))


def _validate_build_profile(profile: ModelProfile, corpus: Trace2SkillCorpus) -> None:
    if profile.memory is not None:
        raise ValueError("Trace2Skill build requires memory=null")
    if profile.skill is None or profile.skill.framework != "trace2skill":
        raise ValueError("Trace2Skill build requires a profile-bound trace2skill skill path")
    collection_profile = corpus.manifest.get("assistant_model_profile")
    if not isinstance(collection_profile, dict):
        raise Trace2SkillError("trajectory collection has no assistant profile provenance")
    expected_hash = collection_profile.get("behavior_sha256")
    actual_hash = _profile_behavior_sha256(profile)
    if expected_hash != actual_hash:
        raise ValueError(
            "assistant model behavior does not match the trajectory collection; "
            "collect a separate corpus for this model/profile configuration"
        )


def _prepare_collection_directory(
    *,
    output_root: Path,
    resume_dir: Path | None,
    prepared: PreparedPipeline,
    trace_config: Trace2SkillConfig,
    dataset_info: EvolutionDatasetInfo,
    samples: list[Sample],
    profile_path: str,
    collection_contract: Trace2SkillCollectionContract,
) -> tuple[Path, str]:
    settings = trace_config.run
    profile = load_model_profile(profile_path)
    profile_source = Path(profile_path).expanduser().resolve()
    simulator_model_files = {
        key: {
            "path": str(Path(path).expanduser().resolve()),
            "sha256": _sha256_file(path),
        }
        for key, path in prepared.simulator_config.models.items()
    }
    simulator_prompt_files = {
        key: {
            "path": str(Path(path).expanduser().resolve()),
            "sha256": _sha256_file(path),
        }
        for key, path in {
            "controller": prepared.simulator_config.prompts.controller,
            "satisfaction": prepared.simulator_config.prompts.satisfaction,
            "realizer_clear": prepared.simulator_config.prompts.realizer_clear,
            "realizer_abstract": prepared.simulator_config.prompts.realizer_abstract,
        }.items()
    }
    manifest_basis = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COLLECTION_ARTIFACT_TYPE,
        "method": settings.method,
        "initial_skill": {"label": NO_SKILL, "content": "", "sha256": _hash_text("")},
        "collection_contract": {
            **collection_contract.to_dict(),
            "turn_budget": settings.turn_budget,
            "seed": settings.seed,
            "infrastructure_max_attempts": settings.infrastructure_max_attempts,
            "retry_policy": "infrastructure_only",
            "distillation_calls": False,
        },
        "development_dataset": dataset_info.to_dict(),
        "assistant_model_profile": {
            "source_path": str(profile_source),
            "source_sha256": _sha256_file(profile_source),
            "settings": profile.model_dump(mode="json"),
            "behavior": _profile_behavior_snapshot(profile),
            "behavior_sha256": _profile_behavior_sha256(profile),
            "profile_skill_enabled_during_collection": False,
        },
        "assistant_rollout_config": {
            "assistant": prepared.assistant_config.components.assistant,
            "baseline": prepared.assistant_config.components.baseline,
            "profile_skill_enabled": prepared.assistant_config.components.profile_skill_enabled,
            "memory_updates": False,
        },
        "simulator": {
            "config": prepared.simulator_config.model_dump(mode="json"),
            "model_profile_files": simulator_model_files,
            "prompt_files": simulator_prompt_files,
        },
        "sample_order_sha256": _hash_text("\n".join(item.sample_id for item in samples)),
        "git_commit": read_git_commit(),
    }
    fingerprint = _hash_text(_canonical_json(manifest_basis))
    if resume_dir is not None:
        run_dir = resume_dir.expanduser().resolve()
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("artifact_type") != COLLECTION_ARTIFACT_TYPE:
            raise ValueError("resume directory is not a trajectory collection")
        if manifest.get("immutable_fingerprint") != fingerprint:
            raise ValueError("resume directory was created with different collection inputs")
        return run_dir, cast(str, manifest["created_at"])

    run_prefix = (
        "trace2skill_trajectories"
        if collection_contract.canonical
        else f"trace2skill_trajectories_trial_n{collection_contract.selected_sample_count}"
    )
    run_id = run_prefix + "_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id += "_" + uuid.uuid4().hex[:8]
    run_dir = output_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(UTC).isoformat()
    manifest = {
        **manifest_basis,
        "run_id": run_id,
        "created_at": created_at,
        "immutable_fingerprint": fingerprint,
        "sample_ids": [item.sample_id for item in samples],
        "visibility_contract": {
            "sealed_model_input": [
                "ordered_visible_user_assistant_dialogue",
                "SUCCESS_or_FAILURE_TURN_LIMIT",
            ],
            "forbidden_for_skill_build": [
                "reason_dag",
                "difficulty",
                "controller_or_satisfaction_state",
                "simulator_prompts",
                "hidden_reasoning",
                "sample_or_user_metadata",
                "final_state",
                "evaluator_explanations",
                "test_data",
            ],
        },
        "artifact_retention": {
            "episode_attempts": "all_attempt_directories_including_infrastructure_failures",
            "trajectory_slots": "one_atomic_whitelisted_slot_per_source_sample",
            "corpus_seal": "corpus.json_after_all_selected_slots_validate",
            "analysis": "forbidden_in_collection_directory",
            "merge": "forbidden_in_collection_directory",
            "skill": "forbidden_in_collection_directory",
            "trial_policy": (
                "not_applicable"
                if collection_contract.canonical
                else "audit_only_and_never_skill_build_source"
            ),
        },
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    if not collection_contract.canonical:
        _atomic_text(
            run_dir / "COLLECTION_TRIAL_ONLY.md",
            "# Non-canonical Trace2Skill collection trial\n\n"
            f"This collection contains only the first "
            f"{collection_contract.selected_sample_count} of "
            f"{collection_contract.source_dataset_sample_count} source-ordered development "
            "samples.\n\n"
            "It retains the collection-stage audit artifacts, but it must not be used as "
            "input to Trace2Skill build, published as a model skill, or used for test-set "
            "evaluation. Run collect without --limit to create the canonical corpus.\n",
        )
    return run_dir, created_at


def _trajectory_entry_hash(
    *,
    dataset_index: int,
    sample_id: str,
    dataset_sha256: str,
    trajectory: VisibleTrajectory,
) -> str:
    whitelisted_record = {
        "dataset_index": dataset_index,
        "sample_id": sample_id,
        "dataset_sha256": dataset_sha256,
        "visible_trajectory": trajectory.model_dump(mode="json"),
    }
    return _hash_text(_canonical_json(whitelisted_record))


def _seal_trajectory_corpus(
    run_dir: Path,
    *,
    slots: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    sample_ids = manifest.get("sample_ids")
    dataset = manifest.get("development_dataset")
    collection_contract = _normalize_collection_contract(manifest)
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != collection_contract.selected_sample_count
    ):
        raise Trace2SkillError("cannot seal corpus with invalid sample order")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("sha256"), str):
        raise Trace2SkillError("cannot seal corpus without a dataset hash")
    if len(slots) != len(sample_ids):
        raise Trace2SkillError("cannot seal an incomplete trajectory corpus")
    entry_hashes: list[str] = []
    for index, (slot, sample_id) in enumerate(zip(slots, sample_ids, strict=True)):
        if (
            slot.get("artifact_type") != COLLECTION_SLOT_ARTIFACT_TYPE
            or slot.get("status") != "complete"
            or slot.get("dataset_index") != index
            or slot.get("sample_id") != sample_id
            or slot.get("dataset_sha256") != dataset["sha256"]
        ):
            raise Trace2SkillError(f"cannot seal invalid trajectory slot {index}")
        trajectory = VisibleTrajectory.model_validate(slot.get("visible_trajectory"))
        if (
            slot.get("outcome") != trajectory.outcome
            or slot.get("turns") != trajectory.assistant_turns
        ):
            raise Trace2SkillError(
                f"cannot seal trajectory slot with inconsistent outcome: {index}"
            )
        entry_hash = _trajectory_entry_hash(
            dataset_index=index,
            sample_id=cast(str, sample_id),
            dataset_sha256=cast(str, dataset["sha256"]),
            trajectory=trajectory,
        )
        if slot.get("trajectory_sha256") != entry_hash:
            raise Trace2SkillError(f"cannot seal trajectory slot with hash mismatch: {index}")
        entry_hashes.append(entry_hash)
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COLLECTION_ARTIFACT_TYPE,
        "status": "sealed",
        "collection_fingerprint": manifest["immutable_fingerprint"],
        "sample_count": len(slots),
        "run_mode": collection_contract.run_mode,
        "canonical": collection_contract.canonical,
        "skill_build_allowed": collection_contract.skill_build_allowed,
        "source_dataset_sample_count": collection_contract.source_dataset_sample_count,
        "dataset_sha256": dataset["sha256"],
        "sample_order_sha256": manifest["sample_order_sha256"],
        "assistant_behavior_sha256": manifest["assistant_model_profile"]["behavior_sha256"],
        "trajectory_sha256": entry_hashes,
        "corpus_sha256": _hash_text(_canonical_json(entry_hashes)),
        "sealed_at": datetime.now(UTC).isoformat(),
    }
    path = run_dir / "corpus.json"
    if path.exists():
        existing = _read_json(path)
        for key, value in record.items():
            if key != "sealed_at" and existing.get(key) != value:
                raise Trace2SkillError("existing trajectory corpus seal is incompatible")
        return existing
    _atomic_json(path, record)
    return record


def _write_collection_summary(
    run_dir: Path,
    *,
    created_at: str,
    complete: bool,
    slots: list[dict[str, Any]],
    collection_contract: Trace2SkillCollectionContract,
    corpus_sha256: str | None = None,
) -> None:
    pending = [item for item in slots if item.get("status") != "complete"]
    recovery_grants = [
        grant
        for item in slots
        for grant in item.get("infrastructure_recovery_grants", [])
        if isinstance(grant, dict)
    ]
    slots_with_recovery = sum(bool(item.get("infrastructure_recovery_grants")) for item in slots)
    exhausted_pending_count = sum(
        item.get("exhausted_infrastructure_attempts") is True for item in pending
    )
    if not complete:
        next_step = (
            "resume collect with --resume <collection-dir> --retry-exhausted; "
            "Skill build remains blocked until corpus.json is sealed"
            if exhausted_pending_count
            else "resume collect with --resume <collection-dir>; Skill build remains blocked "
            "until corpus.json is sealed"
        )
    elif collection_contract.skill_build_allowed:
        next_step = "run trace2skill build with this collection directory"
    else:
        next_step = "inspect collection artifacts, then run collect without --limit"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COLLECTION_ARTIFACT_TYPE,
        "stage": "collect",
        "created_at": created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "complete" if complete else "incomplete",
        "sealed": complete,
        "run_mode": collection_contract.run_mode,
        "canonical": collection_contract.canonical,
        "skill_build_allowed": collection_contract.skill_build_allowed,
        "collection_contract": collection_contract.to_dict(),
        "valid_trajectory_count": len(slots) - len(pending),
        "pending_count": len(pending),
        "pending_dataset_indices": [item["dataset_index"] for item in pending],
        "corpus_sha256": corpus_sha256,
        "build_ready": complete and collection_contract.skill_build_allowed,
        "distillation_call_count": 0,
        "infrastructure_recovery": {
            "policy": "explicit_retry_exhausted_resume",
            "grant_count": len(recovery_grants),
            "slot_count": slots_with_recovery,
            "additional_attempts_authorized": sum(
                int(grant.get("additional_attempts", 0)) for grant in recovery_grants
            ),
            "recovered_slot_count": sum(
                item.get("status") == "complete"
                and bool(item.get("infrastructure_recovery_grants"))
                for item in slots
            ),
            "exhausted_pending_count": exhausted_pending_count,
        },
        "artifact_inventory": {
            "manifest": (run_dir / "manifest.json").exists(),
            "trajectory_slot_count": len(list((run_dir / "trajectory_slots").glob("*.json"))),
            "episode_attempt_directory_count": sum(
                path.is_dir() for path in run_dir.glob("trajectory_*_attempt_*")
            ),
            "corpus_seal": (run_dir / "corpus.json").exists(),
            "analysis_slot_count": len(list((run_dir / "trajectory_analysis").glob("*.json"))),
            "map_patch_count": len(list((run_dir / "map_patches").glob("*.json"))),
            "merge_artifact_count": len(
                list((run_dir / "merge_levels").glob("level_*/group_*.json"))
            ),
            "final_patch": (run_dir / "final_patch.json").exists(),
            "translated_final_patch": (run_dir / "translated_final_patch.json").exists(),
            "verification_round_count": len(list((run_dir / "verification").glob("round_*.json"))),
            "apply": (run_dir / "apply.json").exists(),
            "skill": (run_dir / "SKILL.md").exists() or (run_dir / "SMOKE_SKILL.md").exists(),
        },
        "intended_use": (
            "skill_build_source"
            if collection_contract.skill_build_allowed
            else "collection_pipeline_audit_only"
        ),
        "next_step": next_step,
    }
    _atomic_json(run_dir / "summary.json", summary)


def _validate_trace_run(
    prepared: PreparedPipeline,
    samples: list[Sample],
    trace_config: Trace2SkillConfig,
    dataset_info: EvolutionDatasetInfo,
) -> None:
    settings = trace_config.run
    if prepared.assistant_config.components.baseline != "base":
        raise ValueError("Trace2Skill development rollouts require baseline=base (No Skill)")
    profile = load_model_profile(prepared.assistant_config.models["assistant"])
    if profile.memory is not None:
        raise ValueError("Trace2Skill development rollouts require memory=null")
    if profile.skill is not None and prepared.assistant_config.components.profile_skill_enabled:
        raise ValueError(
            "Trace2Skill development rollouts must disable the profile-bound skill and start "
            "from No Skill"
        )
    if prepared.simulator_config.policy.max_turns != settings.turn_budget:
        raise ValueError(
            f"Trace2Skill requires simulator max_turns={settings.turn_budget}, found "
            f"{prepared.simulator_config.policy.max_turns}"
        )
    configured_dataset = Path(trace_config.dataset.path).expanduser().resolve()
    if prepared.dataset.path.expanduser().resolve() != dataset_info.path:
        raise ValueError("prepared pipeline is not using the validated development dataset")
    if configured_dataset != dataset_info.path:
        raise ValueError("Trace2Skill config is not using the validated development dataset")
    expected = trace_config.dataset.expected_sample_count
    if len(samples) != expected:
        raise ValueError(f"Trace2Skill requires exactly {expected} development samples")
    if tuple(sample.sample_id for sample in samples) != dataset_info.sample_ids:
        raise ValueError("Trace2Skill samples must be the complete source-ordered dev dataset")
    if len(dataset_info.sample_ids) != expected:
        raise ValueError("validated development dataset count does not match Trace2Skill config")


def _resolve_collection_contract(
    *,
    expected_sample_count: int,
    limit: int | None,
) -> Trace2SkillCollectionContract:
    if limit is None:
        return Trace2SkillCollectionContract(
            run_mode="canonical",
            canonical=True,
            skill_build_allowed=True,
            selected_sample_count=expected_sample_count,
            source_dataset_sample_count=expected_sample_count,
            selection="complete_source_order",
            limit=None,
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if limit >= expected_sample_count:
        raise ValueError(
            "collection limit must be smaller than the canonical development sample count; "
            "omit it to collect the formal 1,000-sample corpus"
        )
    return Trace2SkillCollectionContract(
        run_mode="smoke",
        canonical=False,
        skill_build_allowed=False,
        selected_sample_count=limit,
        source_dataset_sample_count=expected_sample_count,
        selection="source_order_prefix",
        limit=limit,
    )


def _normalize_collection_contract(
    manifest: dict[str, Any],
) -> Trace2SkillCollectionContract:
    raw = manifest.get("collection_contract")
    dataset = manifest.get("development_dataset")
    if not isinstance(raw, dict):
        raise Trace2SkillError("trajectory collection has no collection contract")
    if not isinstance(dataset, dict):
        raise Trace2SkillError("trajectory collection has no development dataset provenance")
    dataset_count = dataset.get("sample_count")
    selected_count = raw.get("selected_sample_count")
    source_count = raw.get("source_dataset_sample_count", dataset_count)
    if (
        not isinstance(dataset_count, int)
        or isinstance(dataset_count, bool)
        or dataset_count < 1
        or not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or selected_count < 1
        or not isinstance(source_count, int)
        or isinstance(source_count, bool)
        or source_count != dataset_count
        or selected_count > source_count
    ):
        raise Trace2SkillError("trajectory collection has an invalid sample-count contract")

    selection = raw.get("selection")
    limit = raw.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise Trace2SkillError("trajectory collection has an invalid limit")
    inferred_canonical = (
        selected_count == source_count and selection == "complete_source_order" and limit is None
    )
    run_mode = raw.get("run_mode", "canonical" if inferred_canonical else "smoke")
    canonical = raw.get("canonical", inferred_canonical)
    skill_build_allowed = raw.get("skill_build_allowed", inferred_canonical)
    if run_mode not in ("canonical", "smoke"):
        raise Trace2SkillError("trajectory collection has an invalid run mode")
    if not isinstance(canonical, bool) or not isinstance(skill_build_allowed, bool):
        raise Trace2SkillError("trajectory collection has invalid canonical-use flags")

    if inferred_canonical:
        valid = run_mode == "canonical" and canonical and skill_build_allowed
    else:
        valid = (
            selected_count < source_count
            and selection == "source_order_prefix"
            and limit == selected_count
            and run_mode == "smoke"
            and not canonical
            and not skill_build_allowed
        )
    if not valid:
        raise Trace2SkillError("trajectory collection contract is internally inconsistent")
    return Trace2SkillCollectionContract(
        run_mode=cast(Trace2SkillRunMode, run_mode),
        canonical=canonical,
        skill_build_allowed=skill_build_allowed,
        selected_sample_count=selected_count,
        source_dataset_sample_count=source_count,
        selection=cast(
            Literal["complete_source_order", "source_order_prefix"],
            selection,
        ),
        limit=cast(int | None, limit),
    )


def _resolve_run_contract(
    *,
    expected_sample_count: int,
    limit: int | None,
) -> Trace2SkillRunContract:
    if limit is None:
        return Trace2SkillRunContract(
            run_mode="canonical",
            canonical=True,
            evaluation_allowed=True,
            selected_sample_count=expected_sample_count,
            selection="complete_source_order",
            limit=None,
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if limit >= expected_sample_count:
        raise ValueError(
            "limit must be smaller than the canonical development sample count; "
            "omit it to run the formal 1,000-sample method"
        )
    return Trace2SkillRunContract(
        run_mode="smoke",
        canonical=False,
        evaluation_allowed=False,
        selected_sample_count=limit,
        selection="source_order_prefix",
        limit=limit,
    )


def _read_slot(path: Path, *, dataset_index: int, sample_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _read_json(path)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise Trace2SkillError(f"unsupported trajectory slot schema: {path}")
    if value.get("dataset_index") != dataset_index or value.get("sample_id") != sample_id:
        raise Trace2SkillError(f"trajectory slot identity mismatch: {path}")
    return value


def _record_uncaught_attempt(
    *,
    sample_id: str,
    run_dir: Path,
    run_id: str,
    error: Exception,
) -> RunResult:
    target = run_dir / run_id
    if not target.exists():
        return record_setup_failure(
            sample_id=sample_id,
            output_dir=run_dir,
            error=error,
            run_id=run_id,
            config_snapshot={"mode": "trace2skill_visible_parallel"},
        )
    return RunResult(
        sample_id=sample_id,
        status="failed",
        turns=0,
        run_dir=target,
        error=f"{type(error).__name__}: {error}",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Trace2SkillError(f"cannot read Trace2Skill artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Trace2SkillError(f"Trace2Skill artifact must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _messages_hash(messages: list[dict[str, str]]) -> str:
    return _hash_text(_canonical_json(messages))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
