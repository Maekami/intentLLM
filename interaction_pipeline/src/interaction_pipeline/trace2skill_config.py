import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from interaction_pipeline.evolution_config import EvolutionDatasetSettings

DEFAULT_TRACE2SKILL_CONFIG = Path(__file__).resolve().parents[2] / "configs/trace2skill.yaml"


class Trace2SkillRunSettings(BaseModel):
    """Frozen controls for Trace2Skill-Visible-Combined-Parallel-B32."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["Trace2Skill-Visible-Combined-Parallel-B32"] = (
        "Trace2Skill-Visible-Combined-Parallel-B32"
    )
    initial_skill: Literal["No Skill"] = "No Skill"
    information_boundary: Literal["visible_dialogue_plus_binary_outcome"] = (
        "visible_dialogue_plus_binary_outcome"
    )
    analysis_mode: Literal["combined"] = "combined"
    map_batch_size: Literal[1] = 1
    merge_batch_size: Literal[32] = 32
    turn_budget: Literal[20] = 20
    trajectory_concurrency: int = Field(default=32, ge=1)
    merge_concurrency: int = Field(default=8, ge=1, le=32)
    infrastructure_max_attempts: Literal[3] = 3
    # Frozen to the official parallel evolver defaults. Provider-level retries
    # still come from the selected model profile.
    max_continuations: Literal[2] = 2
    format_fix_rounds: Literal[2] = 2
    max_merge_levels: Literal[5] = 5
    max_verification_rounds: Literal[3] = 3
    max_skill_lines: Literal[500] = 500
    seed: int = 42
    output_dir: str = "runs_trace2skill"
    group_output_by_profile: bool = True

    @model_validator(mode="after")
    def validate_parallelism(self) -> Self:
        if self.merge_concurrency > self.merge_batch_size:
            raise ValueError("merge_concurrency must not exceed merge_batch_size")
        return self


class Trace2SkillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: EvolutionDatasetSettings = Field(default_factory=EvolutionDatasetSettings)
    run: Trace2SkillRunSettings = Field(default_factory=Trace2SkillRunSettings)

    @model_validator(mode="after")
    def validate_frozen_development_shape(self) -> Self:
        expected_counts = {"easy": 334, "medium": 333, "hard": 333}
        if self.dataset.expected_sample_count != 1000:
            raise ValueError("Trace2Skill B32 requires exactly 1,000 development samples")
        if self.dataset.expected_difficulty_counts != expected_counts:
            raise ValueError(
                "Trace2Skill requires frozen difficulty counts easy=334, medium=333, hard=333"
            )
        return self


def load_trace2skill_config(
    custom_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> Trace2SkillConfig:
    raw = _read_yaml(DEFAULT_TRACE2SKILL_CONFIG)
    if custom_path is not None:
        raw = _deep_merge(raw, _read_yaml(Path(custom_path).expanduser().resolve()))
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Trace2SkillConfig.model_validate(raw)


def resolve_trace2skill_output_directory(
    output_root: str | Path,
    *,
    profile_name: str,
    group_by_profile: bool,
    limit: int | None = None,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 999
    ):
        raise ValueError("limit must be an integer from 1 through 999")
    trial_suffix = f"_smoke_n{limit}" if limit is not None else ""
    if not group_by_profile:
        return root / f"trace2skill{trial_suffix}" if limit is not None else root
    profile_label = (
        profile_name if profile_name.endswith("_trace2skill") else f"{profile_name}_trace2skill"
    )
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", profile_label).strip("._")
    if not safe_label:
        raise ValueError("assistant profile_name cannot produce an empty output directory name")
    return root / f"{safe_label}{trial_suffix}"


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
