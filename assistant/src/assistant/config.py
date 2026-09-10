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
    query_max_characters: int = Field(default=2048, ge=1)
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", min_length=1)
    device: str | None = None
    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)


class MemoryContextSettings(BaseModel):
    """Controls which parts of an experience are synthesized into context."""

    model_config = ConfigDict(extra="forbid")

    include_trajectory: bool = True
    include_feedback: bool = True
    max_characters: int = Field(default=8000, ge=1)


class ReMemSettings(BaseModel):
    """Settings specific to the ReMem Think-Refine-Act loop."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=10, ge=1)
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


class SkillSettings(BaseModel):
    """Optional static skill bound one-to-one to an assistant model profile."""

    model_config = ConfigDict(extra="forbid")

    framework: Literal["trace2skill"] = "trace2skill"
    path: Path


GP_ERROR_CODES = frozenset({
    "call.timeout", "call.rate_limit", "call.transient", "output.empty",
    "output.truncated", "output.parse", "output.schema", "contract.reference",
    "contract.coverage", "contract.dependency", "contract.request", "contract.action",
    "context.missing", "context.capacity", "realization.unrealizable", "runtime.io",
})
GP_ROLES = frozenset({"tracker", "intra", "inter", "joint", "generator", "runtime"})


class GPPolicySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_budget: int = Field(default=1, ge=0)
    adjacent_candidate_limit: int = Field(default=2, ge=0)
    max_adjacent_deliveries: int = Field(default=1, ge=0)


class GPRoleBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_input_tokens: int | None = Field(default=None, ge=1)
    expanded_input_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if (self.expanded_input_tokens is not None and self.max_input_tokens is not None
                and self.expanded_input_tokens < self.max_input_tokens):
            raise ValueError("expanded_input_tokens must be >= max_input_tokens")
        return self


class GPContextSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hard_context_tokens: int | None = Field(default=None, ge=1)
    role_budgets: dict[str, GPRoleBudget] = Field(default_factory=dict)
    estimator: Literal["provider_text_with_overhead", "utf8_upper_bound"] = (
        "provider_text_with_overhead"
    )
    chat_overhead_tokens: int = Field(default=128, ge=0)
    preserve_required_materials: Literal[True] = True

    @model_validator(mode="after")
    def roles(self) -> Self:
        if self.role_budgets.keys() - (GP_ROLES - {"runtime"}):
            raise ValueError("unknown GP context role")
        return self


class GPRecoverySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_max_retries: int = Field(default=3, ge=0, le=3)
    by_role_and_code: dict[str, dict[str, int]] = Field(default_factory=dict)
    call_timeout_seconds: float = Field(default=120, gt=0)
    initial_backoff_seconds: float = Field(default=1, ge=0)
    maximum_backoff_seconds: float = Field(default=8, ge=0)
    output_growth_factor: float = Field(default=1.5, ge=1)
    correction_max_characters: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def limits(self) -> Self:
        for role, overrides in self.by_role_and_code.items():
            if role not in GP_ROLES:
                raise ValueError(f"unknown GP recovery role: {role}")
            for code, value in overrides.items():
                if code not in GP_ERROR_CODES or type(value) is not int or not 0 <= value <= 3:
                    raise ValueError("GP retries require a registered code and a limit in 0..3")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum_backoff_seconds must be >= initial_backoff_seconds")
        return self


class GPRealizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_format: Literal["json_units"] = "json_units"
    request_rendering: Literal["frozen_block"] = "frozen_block"
    separator: str = "\n\n"


class GoalProgressionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: Literal["full", "no_tracker", "no_intra", "no_inter", "joint", "no_anticipate"]
    architecture_version: Literal["v2_contracts"] = "v2_contracts"
    prompts: dict[str, Path]
    # Optional legacy alias for parse/schema only, never an extra retry loop.
    format_retries: int | None = Field(default=None, ge=0, le=3)
    turn_timeout_seconds: float = Field(default=300, gt=0)
    max_in_flight_requests: int = Field(default=2, ge=1)
    # Provider-side constrained decoding is separate from mandatory local contracts.
    structured_decoding: dict[
        Literal["tracker", "intra", "inter", "joint", "generator"],
        Literal["schema", "prompt"],
    ] = Field(default_factory=dict)
    policy: GPPolicySettings = Field(default_factory=GPPolicySettings)
    context: GPContextSettings = Field(default_factory=GPContextSettings)
    recovery: GPRecoverySettings = Field(default_factory=GPRecoverySettings)
    realization: GPRealizationSettings = Field(default_factory=GPRealizationSettings)

    def retry_limit(self, role: str, code: str) -> int:
        overrides = self.recovery.by_role_and_code.get(role, {})
        if code in overrides:
            return overrides[code]
        if self.format_retries is not None and code in {"output.parse", "output.schema"}:
            return self.format_retries
        return self.recovery.default_max_retries

    @property
    def roles(self) -> tuple[str, ...]:
        if self.variant == "joint":
            return ("tracker", "joint", "generator")
        return tuple(
            role
            for role in ("tracker", "intra", "inter", "generator")
            if self.variant != f"no_{role}"
        )

    @model_validator(mode="after")
    def validate_prompts(self) -> Self:
        unknown = set(self.prompts) - {"tracker", "intra", "inter", "joint", "generator"}
        missing = set(self.roles) - self.prompts.keys()
        if unknown or missing:
            raise ValueError(
                f"invalid GP prompts: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return self


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
    # A profile-bound skill is activated on top of the unprompted base setting.
    skill: SkillSettings | None = None
    goal_progression: GoalProgressionSettings | None = None

    @model_validator(mode="after")
    def validate_assistant_generation(self) -> Self:
        if "assistant" not in self.generation:
            raise ValueError("model profile must define generation.assistant")
        if self.memory is not None and self.skill is not None:
            raise ValueError("a model profile cannot enable memory and a static skill together")
        if self.goal_progression is not None:
            if self.memory is not None or self.skill is not None:
                raise ValueError("goal_progression cannot be combined with memory or skill")
            for role in self.goal_progression.roles:
                key = "assistant" if role == "generator" else role
                if key not in self.generation:
                    raise ValueError(f"goal_progression requires generation.{key}")
                if any(value is None for value in self.generation[key].model_dump().values()):
                    raise ValueError(f"goal_progression requires complete generation.{key}")
        return self


class ComponentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant: str = "llm_assistant"
    baseline: str = "base"
    # Trace2Skill authoring disables this so all development rollouts use No Skill.
    profile_skill_enabled: bool = True


class InteractCompActionGuardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # False is the compatibility default for historical resolved snapshots.
    # The shipped assistant.yaml explicitly enables the current baseline.
    enabled: bool = False
    # InteractComp routes AskNL validation through its responder/user model and
    # overrides that model's temperature to 1.0. The pipeline supplies the
    # simulator model client; these settings are therefore intentionally shared
    # across every tested assistant profile.
    generation: GenerationSettings = Field(
        default_factory=lambda: GenerationSettings(
            temperature=1.0,
            top_p=1.0,
            max_completion_tokens=2048,
        )
    )
    # Official AskNL validation makes one independent decision per action.
    # These compatibility switches stay configurable but are disabled in the
    # official-aligned baseline.
    confirm_rejections: bool = False
    cache_verdicts: bool = False
    invalid_output_retries: int = Field(default=0, ge=0)


class InteractCompReActSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Official InteractComp permits ten action rounds per task. In this
    # non-terminal dialogue adaptation those rounds are local semantic slots
    # inside one simulator-visible turn; only an accepted action is published.
    semantic_action_budget_per_turn: int = Field(default=10, ge=1)
    # Matches official max_invalid_retries=3: first format attempt plus at most
    # three retries, without advancing the semantic action slot.
    format_retries_per_slot: int = Field(default=3, ge=0)
    action_guard: InteractCompActionGuardSettings = Field(
        default_factory=InteractCompActionGuardSettings
    )


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_base: str
    interactcomp_react: str = "configs/prompts/interactcomp_react.yaml"
    interactcomp_action_guard: str = "configs/prompts/interactcomp_action_guard.yaml"
    trace2skill: str = "configs/skills/trace2skill/SKILL.md"


class AssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: ComponentSettings
    prompts: PromptSettings
    models: dict[str, str]
    interactcomp_react: InteractCompReActSettings = Field(default_factory=InteractCompReActSettings)

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
    if profile.goal_progression is not None:
        settings = profile.goal_progression.model_copy(
            update={
                "prompts": {
                    role: prompt if prompt.is_absolute() else (path.parent / prompt).resolve()
                    for role, prompt in profile.goal_progression.prompts.items()
                }
            }
        )
        profile = profile.model_copy(update={"goal_progression": settings})
    if profile.memory is not None and not profile.memory.path.is_absolute():
        memory = profile.memory.model_copy(
            update={"path": (path.parent / profile.memory.path).resolve()}
        )
        profile = profile.model_copy(update={"memory": memory})
    if profile.skill is not None and not profile.skill.path.is_absolute():
        skill = profile.skill.model_copy(
            update={"path": (path.parent / profile.skill.path).resolve()}
        )
        profile = profile.model_copy(update={"skill": skill})
    return profile
