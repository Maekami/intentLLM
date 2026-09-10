import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_EVOLUTION_CONFIG = Path(__file__).resolve().parents[2] / "configs/evolution.yaml"


class EvolutionRunSettings(BaseModel):
    """Run controls that are intentionally isolated from test batch settings."""

    model_config = ConfigDict(extra="forbid")

    mini_batch_size: int = Field(default=8, ge=1)
    concurrency: int = Field(default=8, ge=1)
    sample_retries: int = Field(default=0, ge=0)
    seed: int = 42
    max_turns: int | None = Field(default=None, ge=1)
    output_dir: str = "runs_evolution"
    group_output_by_profile: bool = True

    @model_validator(mode="after")
    def validate_concurrency(self) -> Self:
        if self.concurrency > self.mini_batch_size:
            raise ValueError("evolution concurrency must not exceed mini_batch_size")
        return self


class EvolutionDatasetSettings(BaseModel):
    """Frozen development dataset and its offline difficulty assignment."""

    model_config = ConfigDict(extra="forbid")

    path: str = "../user_simulator/dataset/DAG_evolution_dev.jsonl"
    original_path: str = "../user_simulator/dataset/DAG_evolution_dev_original.jsonl"
    metadata_path: str = "../user_simulator/dataset/DAG_evolution_dev_metadata.json"
    difficulty_field: str = Field(default="difficulty", min_length=1)
    expected_sample_count: int = Field(default=1000, ge=1)
    expected_difficulty_counts: dict[Literal["easy", "medium", "hard"], int] = Field(
        default_factory=lambda: {"easy": 334, "medium": 333, "hard": 333}
    )

    @model_validator(mode="after")
    def validate_expected_counts(self) -> Self:
        if set(self.expected_difficulty_counts) != {"easy", "medium", "hard"}:
            raise ValueError("expected_difficulty_counts must contain easy, medium, and hard")
        if any(count < 0 for count in self.expected_difficulty_counts.values()):
            raise ValueError("expected difficulty counts must be non-negative")
        if sum(self.expected_difficulty_counts.values()) != self.expected_sample_count:
            raise ValueError("expected difficulty counts must sum to expected_sample_count")
        return self


class EvolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: EvolutionDatasetSettings = Field(default_factory=EvolutionDatasetSettings)
    run: EvolutionRunSettings = Field(default_factory=EvolutionRunSettings)


def load_evolution_config(
    custom_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> EvolutionConfig:
    raw = _read_yaml(DEFAULT_EVOLUTION_CONFIG)
    if custom_path is not None:
        raw = _deep_merge(raw, _read_yaml(Path(custom_path).expanduser().resolve()))
    if overrides:
        raw = _deep_merge(raw, overrides)
    return EvolutionConfig.model_validate(raw)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_profile_output_directory(
    output_root: str | Path,
    *,
    profile_name: str,
    memory_framework: Literal["exprag", "remem"],
    group_by_profile: bool,
) -> Path:
    """Return an isolated model+method output root for one evolution stream."""

    root = Path(output_root).expanduser().resolve()
    if not group_by_profile:
        return root
    suffix = f"_{memory_framework}"
    label = profile_name if profile_name.endswith(suffix) else f"{profile_name}{suffix}"
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    if not safe_label:
        raise ValueError("assistant profile_name cannot produce an empty output directory name")
    return root / safe_label
