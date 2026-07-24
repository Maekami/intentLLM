import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationSettings(BaseModel):
    temperature: float
    max_completion_tokens: int


class RetrySettings(BaseModel):
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 8.0


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_name: str
    provider: str
    model_id: str
    base_url: str
    routing: dict[str, bool]
    reasoning: dict[str, Any]
    generation: dict[str, GenerationSettings]
    structured_output: dict[str, Any]
    retry: RetrySettings


class PolicySettings(BaseModel):
    medium_clear_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    monotonic_satisfaction: bool = True
    auto_expose_backbone_on_empty_queue: bool = True
    enforce_controller_prefix_closure: bool = True
    max_turns: int = Field(default=20, ge=1)


class AuditSettings(BaseModel):
    enabled: bool = True
    level: str = "full"
    output_dir: str = "runs"


class SimulatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = "dataset/DAG.jsonl"
    components: dict[str, str]
    models: dict[str, str]
    policy: PolicySettings
    audit: AuditSettings


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_title: str = "Reason-DAG User Simulator"


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


def load_config(
    custom_path: str | Path | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> SimulatorConfig:
    base = _read_yaml("configs/simulator.yaml")
    if custom_path:
        base = _deep_merge(base, _read_yaml(custom_path))
    # Explicit, narrowly scoped environment overrides.
    if value := os.getenv("REASON_DAG_DATASET_PATH"):
        base["dataset_path"] = value
    if value := os.getenv("REASON_DAG_AUDIT_LEVEL"):
        base.setdefault("audit", {})["level"] = value
    if value := os.getenv("REASON_DAG_MAX_TURNS"):
        base.setdefault("policy", {})["max_turns"] = int(value)
    if cli_overrides:
        base = _deep_merge(base, cli_overrides)
    return SimulatorConfig.model_validate(base)


def load_model_profile(
    name_or_path: str = "deepseek_v4_pro",
    *,
    overrides: dict[str, Any] | None = None,
) -> ModelProfile:
    path = Path(name_or_path)
    if not path.exists():
        path = Path("configs/models") / f"{name_or_path}.yaml"
    raw = _read_yaml(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return ModelProfile.model_validate(raw)
