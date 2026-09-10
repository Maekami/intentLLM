from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulator_root: str = "../user_simulator"
    assistant_root: str = "../assistant"
    simulator_config: str | None = None
    assistant_config: str | None = None


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: Literal["easy", "medium", "hard"] = "medium"
    seed: int = 42
    concurrency: int = Field(default=4, ge=1)
    sample_retries: int = Field(default=3, ge=0)
    update_memory: bool = True
    output_dir: str = "runs"
    audit_level: Literal["summary", "full"] = "full"


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projects: ProjectSettings
    run: RunSettings

    @model_validator(mode="after")
    def validate_project_roots(self) -> Self:
        for label, raw in (
            ("simulator_root", self.projects.simulator_root),
            ("assistant_root", self.projects.assistant_root),
        ):
            path = resolve_pipeline_path(raw)
            if not path.is_dir():
                raise ValueError(f"{label} is not a directory: {path}")
        return self


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
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


def resolve_pipeline_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def load_pipeline_config(
    custom_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    raw = _read_yaml(PROJECT_ROOT / "configs/pipeline.yaml")
    if custom_path is not None:
        raw = _deep_merge(raw, _read_yaml(Path(custom_path).expanduser().resolve()))
    if overrides:
        raw = _deep_merge(raw, overrides)
    return PipelineConfig.model_validate(raw)
