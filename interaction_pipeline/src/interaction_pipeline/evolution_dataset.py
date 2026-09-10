from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from user_simulator.data.loader import DatasetLoader
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty

from interaction_pipeline.evolution_config import EvolutionDatasetSettings

ALLOWED_DIFFICULTIES = {item.value for item in Difficulty}
ASSIGNMENT_ALGORITHM = "sha256_seeded_ranking_v1"


@dataclass(frozen=True)
class EvolutionDatasetInfo:
    path: Path
    sha256: str
    original_path: Path
    original_sha256: str
    metadata_path: Path
    metadata_sha256: str
    sample_ids: tuple[str, ...]
    difficulty_field: str
    difficulty_counts: dict[str, int]
    assignment_seed: int
    assignment_algorithm: str
    ordered_assignments_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "original_path": str(self.original_path),
            "original_sha256": self.original_sha256,
            "metadata_path": str(self.metadata_path),
            "metadata_sha256": self.metadata_sha256,
            "sample_count": len(self.sample_ids),
            "difficulty_field": self.difficulty_field,
            "difficulty_counts": self.difficulty_counts,
            "assignment_seed": self.assignment_seed,
            "assignment_algorithm": self.assignment_algorithm,
            "ordered_assignments_sha256": self.ordered_assignments_sha256,
            "order": "source_jsonl_order",
        }

    @property
    def index_by_sample_id(self) -> dict[str, int]:
        return {sample_id: index for index, sample_id in enumerate(self.sample_ids)}


def validate_evolution_dataset(
    loader: DatasetLoader,
    settings: EvolutionDatasetSettings,
) -> tuple[list[Sample], EvolutionDatasetInfo]:
    """Validate the frozen assignment and return samples in source JSONL order."""

    processed_path = loader.path.expanduser().resolve()
    configured_path = Path(settings.path).expanduser().resolve()
    if processed_path != configured_path:
        raise ValueError(
            f"evolution dataset mismatch: loader uses {processed_path}, config uses {configured_path}"
        )
    original_path = Path(settings.original_path).expanduser().resolve()
    metadata_path = Path(settings.metadata_path).expanduser().resolve()
    processed_rows = _read_jsonl_objects(processed_path)
    original_rows = _read_jsonl_objects(original_path)
    if len(processed_rows) != settings.expected_sample_count:
        raise ValueError(
            f"expected {settings.expected_sample_count} processed samples, "
            f"found {len(processed_rows)}"
        )
    if len(original_rows) != settings.expected_sample_count:
        raise ValueError(
            f"expected {settings.expected_sample_count} original samples, "
            f"found {len(original_rows)}"
        )

    difficulty_field = settings.difficulty_field
    sample_ids: list[str] = []
    difficulties: list[str] = []
    assignment_digest = hashlib.sha256()
    for line_number, (processed, original) in enumerate(
        zip(processed_rows, original_rows, strict=True),
        1,
    ):
        processed_id = processed.get("sample_id")
        original_id = original.get("sample_id")
        if processed_id != original_id:
            raise ValueError(
                f"dataset order mismatch at line {line_number}: "
                f"processed={processed_id!r}, original={original_id!r}"
            )
        if not isinstance(processed_id, str) or not processed_id:
            raise ValueError(f"invalid sample_id at line {line_number}")
        difficulty = processed.get(difficulty_field)
        if difficulty not in ALLOWED_DIFFICULTIES:
            raise ValueError(f"invalid {difficulty_field!r} for {processed_id}: {difficulty!r}")
        base_row = dict(processed)
        base_row.pop(difficulty_field)
        if base_row != original:
            raise ValueError(
                f"processed sample {processed_id!r} differs from the original beyond "
                f"the {difficulty_field!r} field"
            )
        sample_ids.append(processed_id)
        difficulties.append(str(difficulty))
        assignment_digest.update(f"{processed_id}\0{difficulty}\n".encode())

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evolution dataset contains duplicate sample_id values")
    actual_counts = dict(Counter(difficulties))
    expected_counts = dict(settings.expected_difficulty_counts)
    if actual_counts != expected_counts:
        raise ValueError(
            f"difficulty counts mismatch: expected {expected_counts}, found {actual_counts}"
        )

    processed_sha256 = _sha256_file(processed_path)
    original_sha256 = _sha256_file(original_path)
    metadata = _read_json_object(metadata_path)
    expected_metadata = {
        "schema_version": 1,
        "difficulty_field": difficulty_field,
        "sample_count": settings.expected_sample_count,
        "difficulty_counts": expected_counts,
        "ordered_assignments_sha256": assignment_digest.hexdigest(),
        "order": "source_jsonl_order",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"difficulty metadata {key!r} mismatch: expected {expected!r}, "
                f"found {metadata.get(key)!r}"
            )
    if _nested_value(metadata, "processed", "sha256") != processed_sha256:
        raise ValueError("processed dataset SHA-256 does not match difficulty metadata")
    if _nested_value(metadata, "original_copy", "sha256") != original_sha256:
        raise ValueError("original dataset SHA-256 does not match difficulty metadata")
    seed = metadata.get("seed")
    algorithm = metadata.get("algorithm")
    if not isinstance(seed, int):
        raise TypeError("difficulty metadata seed must be an integer")
    if algorithm != ASSIGNMENT_ALGORITHM:
        raise ValueError(
            f"unsupported difficulty assignment algorithm: {algorithm!r}; "
            f"expected {ASSIGNMENT_ALGORITHM!r}"
        )
    reproduced = _seeded_balanced_assignment(sample_ids, seed, expected_counts)
    for sample_id, difficulty in zip(sample_ids, difficulties, strict=True):
        if reproduced[sample_id] != difficulty:
            raise ValueError(
                f"difficulty for {sample_id!r} does not match the declared assignment seed"
            )

    samples = loader.load_all_samples()
    if [sample.sample_id for sample in samples] != sample_ids:
        raise ValueError("validated JSONL order differs from DatasetLoader order")
    return samples, EvolutionDatasetInfo(
        path=processed_path,
        sha256=processed_sha256,
        original_path=original_path,
        original_sha256=original_sha256,
        metadata_path=metadata_path,
        metadata_sha256=_sha256_file(metadata_path),
        sample_ids=tuple(sample_ids),
        difficulty_field=difficulty_field,
        difficulty_counts={key: actual_counts[key] for key in ("easy", "medium", "hard")},
        assignment_seed=seed,
        assignment_algorithm=algorithm,
        ordered_assignments_sha256=assignment_digest.hexdigest(),
    )


def sample_evolution_difficulty(sample: Sample, *, field: str) -> Difficulty:
    extra = sample.model_extra or {}
    value = extra.get(field)
    if value not in ALLOWED_DIFFICULTIES:
        raise ValueError(
            f"sample {sample.sample_id!r} has invalid evolution difficulty "
            f"in field {field!r}: {value!r}"
        )
    return Difficulty(str(value))


def select_samples_in_dataset_order(
    all_samples: list[Sample],
    requested_sample_ids: list[str],
) -> list[Sample]:
    if len(requested_sample_ids) != len(set(requested_sample_ids)):
        raise ValueError("duplicate --sample-id values are not allowed")
    requested = set(requested_sample_ids)
    known = {sample.sample_id for sample in all_samples}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown sample_id values: {unknown}")
    return [sample for sample in all_samples if sample.sample_id in requested]


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number}: JSONL row must be an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evolution dataset {path}: {exc}") from exc
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evolution metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"evolution metadata {path} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash evolution dataset file {path}: {exc}") from exc
    return digest.hexdigest()


def _nested_value(mapping: dict[str, Any], first: str, second: str) -> Any:
    nested = mapping.get(first)
    return nested.get(second) if isinstance(nested, dict) else None


def _seeded_balanced_assignment(
    sample_ids: list[str],
    seed: int,
    counts: dict[str, int],
) -> dict[str, str]:
    def seeded_rank(sample_id: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()

    ranked_ids = sorted(sample_ids, key=lambda sample_id: (seeded_rank(sample_id), sample_id))
    assignment: dict[str, str] = {}
    start = 0
    for difficulty in ("easy", "medium", "hard"):
        end = start + counts[difficulty]
        assignment.update({sample_id: difficulty for sample_id in ranked_ids[start:end]})
        start = end
    return assignment
