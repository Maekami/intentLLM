from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from user_simulator.config import ModelProfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AITR_CONFIG_PATH = PROJECT_ROOT / "configs/aitr.yaml"


class AITRConfig(ModelProfile):
    """Validated AITR judge profile loaded from YAML."""

    prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aitr_contract(self) -> Self:
        if self.provider != "openrouter":
            raise ValueError("AITR judge currently supports only the openrouter provider")
        if "aitr" not in self.generation:
            raise ValueError("AITR config must define generation.aitr")
        if self.retry.max_attempts < 1:
            raise ValueError("retry.max_attempts must be at least 1")
        if self.retry.initial_backoff_seconds < 0:
            raise ValueError("retry.initial_backoff_seconds must be non-negative")
        if self.retry.maximum_backoff_seconds < self.retry.initial_backoff_seconds:
            raise ValueError(
                "retry.maximum_backoff_seconds must be greater than or equal to "
                "retry.initial_backoff_seconds"
            )
        if self.routing.require_parameters is not True:
            raise ValueError("routing.require_parameters must be true for strict AITR output")
        return self

    @property
    def hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AITREnvironmentSettings(BaseSettings):
    """Secrets and OpenRouter request headers kept outside the judge YAML."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_http_referer: str = Field(default="", validation_alias="OPENROUTER_HTTP_REFERER")
    openrouter_app_title: str = Field(
        default="Intent Evaluation Metrics",
        validation_alias="OPENROUTER_APP_TITLE",
    )


def load_aitr_config(
    custom_path: str | Path | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> AITRConfig:
    """Load default YAML, then custom YAML, environment and CLI overrides."""

    raw = _read_yaml(DEFAULT_AITR_CONFIG_PATH)
    if custom_path is not None:
        raw = _deep_merge(raw, _read_yaml(Path(custom_path).expanduser().resolve()))
    raw = _deep_merge(raw, _environment_overrides())
    if cli_overrides:
        raw = _deep_merge(raw, cli_overrides)
    config = AITRConfig.model_validate(raw)
    prompt_path = Path(config.prompt).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path
    prompt_path = prompt_path.resolve()
    if not prompt_path.is_file():
        raise ValueError(f"AITR prompt file does not exist: {prompt_path}")
    return config.model_copy(update={"prompt": str(prompt_path)})


def _environment_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if value := os.getenv("AITR_MODEL"):
        overrides["model_id"] = value
    if value := os.getenv("AITR_REASONING_EFFORT"):
        overrides["reasoning"] = {
            "enabled": value.lower() != "none",
            "effort": value,
        }
    if value := os.getenv("AITR_MAX_COMPLETION_TOKENS"):
        overrides["generation"] = {"aitr": {"max_completion_tokens": int(value)}}
    if value := os.getenv("AITR_MAX_ATTEMPTS"):
        overrides.setdefault("retry", {})["max_attempts"] = int(value)
    if value := os.getenv("AITR_INITIAL_BACKOFF_SECONDS"):
        overrides.setdefault("retry", {})["initial_backoff_seconds"] = float(value)
    if value := os.getenv("AITR_MAXIMUM_BACKOFF_SECONDS"):
        overrides.setdefault("retry", {})["maximum_backoff_seconds"] = float(value)
    if value := os.getenv("AITR_BASE_URL"):
        overrides["base_url"] = value
    return overrides


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"cannot read AITR config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse AITR config {path}: {exc}") from exc
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
