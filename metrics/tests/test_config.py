from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from intent_metrics.config import (
    DEFAULT_AITR_CONFIG_PATH,
    AITREnvironmentSettings,
    load_aitr_config,
)

_AITR_ENVIRONMENT_KEYS = (
    "AITR_MODEL",
    "AITR_REASONING_EFFORT",
    "AITR_MAX_COMPLETION_TOKENS",
    "AITR_MAX_ATTEMPTS",
    "AITR_INITIAL_BACKOFF_SECONDS",
    "AITR_MAXIMUM_BACKOFF_SECONDS",
    "AITR_BASE_URL",
)


@pytest.fixture(autouse=True)
def clear_aitr_environment(monkeypatch) -> None:
    for key in _AITR_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_aitr_config_is_loaded_from_metrics_yaml(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_aitr_config()
    assert DEFAULT_AITR_CONFIG_PATH.is_file()
    assert config.model_id == "openai/gpt-5.6-luna"
    assert config.reasoning.enabled is True
    assert config.reasoning.effort == "high"
    assert config.generation["aitr"].max_completion_tokens == 4096
    assert config.retry.max_attempts == 3
    assert config.structured_output.strict is True
    assert config.prompt.endswith("metrics/configs/prompts/aitr.yaml")


def test_custom_yaml_is_deep_merged_over_default(tmp_path) -> None:
    custom = tmp_path / "custom_aitr.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "model_id": "custom/judge",
                "reasoning": {"effort": "low"},
                "generation": {"aitr": {"max_completion_tokens": 1024}},
                "retry": {"max_attempts": 2},
            }
        ),
        encoding="utf-8",
    )
    config = load_aitr_config(custom)
    assert config.model_id == "custom/judge"
    assert config.reasoning.enabled is True
    assert config.reasoning.effort == "low"
    assert config.generation["aitr"].temperature is None
    assert config.generation["aitr"].max_completion_tokens == 1024
    assert config.retry.max_attempts == 2
    assert config.retry.maximum_backoff_seconds == 8.0


def test_override_precedence_is_default_custom_environment_then_cli(monkeypatch, tmp_path) -> None:
    custom = tmp_path / "custom_aitr.yaml"
    custom.write_text("model_id: custom/judge\n", encoding="utf-8")
    monkeypatch.setenv("AITR_MODEL", "environment/judge")
    monkeypatch.setenv("AITR_REASONING_EFFORT", "none")
    config = load_aitr_config(
        custom,
        cli_overrides={
            "model_id": "cli/judge",
            "reasoning": {"enabled": True, "effort": "medium"},
        },
    )
    assert config.model_id == "cli/judge"
    assert config.reasoning.enabled is True
    assert config.reasoning.effort == "medium"


def test_environment_can_disable_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("AITR_REASONING_EFFORT", "none")
    config = load_aitr_config()
    assert config.reasoning.enabled is False
    assert config.reasoning.effort == "none"


def test_invalid_retry_bounds_are_rejected(tmp_path) -> None:
    custom = tmp_path / "invalid_aitr.yaml"
    custom.write_text(
        "retry:\n  initial_backoff_seconds: 5.0\n  maximum_backoff_seconds: 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="maximum_backoff_seconds"):
        load_aitr_config(custom)


def test_config_hash_changes_with_judge_configuration() -> None:
    first = load_aitr_config()
    second = load_aitr_config(cli_overrides={"model_id": "another/judge"})
    assert len(first.hash) == 64
    assert first.hash != second.hash


def test_api_key_remains_environment_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-fixture")
    environment = AITREnvironmentSettings()
    config = load_aitr_config()
    assert environment.openrouter_api_key == "secret-fixture"
    assert "secret-fixture" not in config.model_dump_json()
    assert "OPENROUTER_API_KEY" not in DEFAULT_AITR_CONFIG_PATH.read_text(encoding="utf-8")
