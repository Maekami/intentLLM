import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_completion_tokens: int = Field(ge=1)


class RetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1)
    initial_backoff_seconds: float = Field(default=1.0, ge=0.0)
    maximum_backoff_seconds: float = Field(default=8.0, ge=0.0)

    @model_validator(mode="after")
    def validate_backoff_bounds(self) -> Self:
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "maximum_backoff_seconds must be greater than or equal to initial_backoff_seconds"
            )
        return self


class LocalChatTemplateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_thinking: bool
    preserve_thinking: bool = False


class ReasoningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    effort: str | None = None
    exclude_from_response: bool = True
    local_chat_template: LocalChatTemplateSettings | None = None

    @model_validator(mode="after")
    def validate_local_thinking_switch(self) -> Self:
        if (
            self.local_chat_template is not None
            and self.local_chat_template.enable_thinking != self.enabled
        ):
            raise ValueError("reasoning.enabled and local_chat_template.enable_thinking must match")
        return self


class MemoryRetrievalSettings(BaseModel):
    """Search configuration for an Evo-Memory profile."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["bm25", "sentence_transformers"] = "bm25"
    top_k: int = Field(default=4, ge=1)
    min_score: float = Field(default=0.0, ge=0.0)
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", min_length=1)
    device: str | None = None
    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)


class MemoryContextSettings(BaseModel):
    """Controls which parts of an experience are synthesized into context."""

    model_config = ConfigDict(extra="forbid")

    include_trajectory: bool = True
    include_feedback: bool = True
    max_characters: int = Field(default=8000, ge=256)


class ReMemSettings(BaseModel):
    """Settings specific to the ReMem Think-Refine-Act loop."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=10, ge=1, le=100)
    enable_pruning: bool = True


class MemorySettings(BaseModel):
    """Optional Evo-Memory framework mounted on an assistant model profile."""

    model_config = ConfigDict(extra="forbid")

    framework: Literal["exprag", "remem"]
    path: Path
    max_entries: int = Field(default=1000, ge=1)
    prune_oldest_when_full: bool = True
    store_successful_only: bool = True
    retrieval: MemoryRetrievalSettings = Field(default_factory=MemoryRetrievalSettings)
    context: MemoryContextSettings = Field(default_factory=MemoryContextSettings)
    remem: ReMemSettings = Field(default_factory=ReMemSettings)


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_name: str
    provider: Literal["openrouter", "vllm", "openai_compatible"]
    model_id: str
    base_url: str
    routing: dict[str, bool] = Field(default_factory=dict)
    reasoning: ReasoningSettings
    generation: dict[str, GenerationSettings]
    retry: RetrySettings
    # `null`/omitted preserves the original memory-free assistant behavior.
    memory: MemorySettings | None = None

    @model_validator(mode="after")
    def validate_assistant_generation(self) -> Self:
        if "assistant" not in self.generation:
            raise ValueError("model profile must define generation.assistant")
        return self


class ComponentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant: str = "llm_assistant"
    baseline: str = "base"


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_base: str


class AssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: ComponentSettings
    prompts: PromptSettings
    models: dict[str, str]

    @model_validator(mode="after")
    def validate_assistant_model(self) -> Self:
        if "assistant" not in self.models:
            raise ValueError("configuration must define models.assistant")
        return self


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_title: str = "Intent Assistant Baselines"
    # vLLM's OpenAI-compatible server accepts any non-empty placeholder when
    # it was launched without --api-key. Set VLLM_API_KEY to the configured
    # server key when authentication is enabled.
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
) -> AssistantConfig:
    base = _read_yaml("configs/assistant.yaml")
    if custom_path:
        base = _deep_merge(base, _read_yaml(custom_path))
    if value := os.getenv("ASSISTANT_BASELINE"):
        base.setdefault("components", {})["baseline"] = value
    if value := os.getenv("ASSISTANT_MODEL_PROFILE"):
        base.setdefault("models", {})["assistant"] = value
    if cli_overrides:
        base = _deep_merge(base, cli_overrides)
    return AssistantConfig.model_validate(base)


def load_model_profile(
    name_or_path: str = "gpt_5_6_luna",
    *,
    overrides: dict[str, Any] | None = None,
) -> ModelProfile:
    path = Path(name_or_path)
    if not path.exists():
        path = Path("configs/models") / f"{name_or_path}.yaml"
    raw = _read_yaml(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    profile = ModelProfile.model_validate(raw)
    if profile.memory is not None and not profile.memory.path.is_absolute():
        memory = profile.memory.model_copy(
            update={"path": (path.parent / profile.memory.path).resolve()}
        )
        profile = profile.model_copy(update={"memory": memory})
    return profile
