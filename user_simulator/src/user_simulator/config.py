import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrySettings(BaseModel):
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 8.0


class LocalChatTemplateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_thinking: bool
    preserve_thinking: bool = False
    reasoning_effort: str | None = Field(default=None, min_length=1)


class ReasoningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    effort: str | None = Field(default="high", min_length=1)
    exclude_from_response: bool = True
    local_chat_template: LocalChatTemplateSettings | None = None

    @model_validator(mode="after")
    def validate_local_thinking_switch(self) -> Self:
        local = self.local_chat_template
        if local is not None and local.enable_thinking != self.enabled:
            raise ValueError("reasoning.enabled and local_chat_template.enable_thinking must match")
        if (
            local is not None
            and local.reasoning_effort is not None
            and self.effort is not None
            and local.reasoning_effort != self.effort
        ):
            raise ValueError(
                "reasoning.effort and local_chat_template.reasoning_effort must match"
            )
        return self


class GenerationSettings(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_completion_tokens: int = Field(ge=1)
    reasoning: ReasoningSettings | None = None


class StructuredOutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"] = "json_schema"
    strict: Literal[True] = True
    require_parameters: Literal[True] = True
    response_healing: bool = False


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_name: str
    provider: Literal["openrouter", "vllm", "openai_compatible"]
    model_id: str
    base_url: str
    routing: dict[str, bool] = Field(default_factory=dict)
    reasoning: ReasoningSettings
    generation: dict[str, GenerationSettings]
    structured_output: StructuredOutputSettings
    retry: RetrySettings


class PolicySettings(BaseModel):
    medium_clear_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    medium_min_nodes: int = Field(default=1, ge=1)
    medium_max_nodes: int | None = Field(default=None, ge=1)
    monotonic_satisfaction: bool = True
    auto_expose_backbone_on_empty_queue: bool = True
    enforce_controller_prefix_closure: bool = True
    max_turns: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_medium_node_bounds(self) -> Self:
        if self.medium_max_nodes is not None and self.medium_max_nodes < self.medium_min_nodes:
            raise ValueError("medium_max_nodes must be greater than or equal to medium_min_nodes")
        return self


class AuditSettings(BaseModel):
    enabled: bool = True
    level: str = "full"
    output_dir: str = "runs"


class ComponentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller: str = "llm_controller"
    satisfaction_updater: str = "llm_satisfaction_updater"
    selection_policy: str = "difficulty_selection"
    realization_policy: str = "difficulty_realization"
    user_realizer: str = "llm_user_realizer"


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller: str
    satisfaction: str
    realizer_clear: str
    realizer_abstract: str


class SimulatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = "dataset/DAG.jsonl"
    components: ComponentSettings
    prompts: PromptSettings
    models: dict[str, str]
    policy: PolicySettings
    audit: AuditSettings


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_title: str = "Reason-DAG User Simulator"
    # vLLM requires a non-empty SDK key even when its local server does not
    # enforce authentication. Override VLLM_API_KEY when the server uses one.
    vllm_api_key: str = "EMPTY"


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
    name_or_path: str = "deepseek_v4_flash_0731",
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
