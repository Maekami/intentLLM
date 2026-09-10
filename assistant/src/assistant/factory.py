from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from assistant.baselines.base import AssistantBaseline, BaseBaseline
from assistant.baselines.interactcomp_action_guard import InteractCompActionGuard
from assistant.baselines.interactcomp_react import (
    InteractCompReActBaseline,
    InteractCompReActSession,
)
from assistant.baselines.prompt_base import PromptBaseBaseline
from assistant.baselines.trace2skill import Trace2SkillBaseline
from assistant.config import (
    AssistantConfig,
    EnvironmentSettings,
    GenerationSettings,
    ModelProfile,
    load_model_profile,
)
from assistant.exceptions import ConfigurationError
from assistant.goal_progression import GoalProgressionBaseline, GoalProgressionSession
from assistant.llm.base import ChatLLMClient
from assistant.llm.openrouter_client import OpenAICompatibleChatClient
from assistant.memory.models import MemoryEntry
from assistant.memory.retrieval import build_retriever
from assistant.memory.sessions import ExpRAGSession, ReMemSession
from assistant.memory.store import JsonMemoryStore
from assistant.prompt import SystemPrompt
from assistant.session import AssistantSession
from assistant.skill import StaticSkill


class PromptArtifact(Protocol):
    name: str
    system: str
    path: str

    @property
    def hash(self) -> str: ...


@dataclass(frozen=True)
class AssistantSpec:
    constructor: type[AssistantSession]


@dataclass(frozen=True)
class BaselineSpec:
    constructor: type[AssistantBaseline]
    prompt_config_key: str | None
    artifact_loader: type[SystemPrompt] | type[StaticSkill] | None = None
    session_constructor: type[AssistantSession] | None = None


@dataclass(frozen=True)
class ResolvedBaseline:
    name: str
    spec: BaselineSpec
    artifact_path: str | None
    source: Literal["assistant_config", "model_profile"]


ASSISTANT_REGISTRY: dict[str, AssistantSpec] = {
    "llm_assistant": AssistantSpec(
        constructor=AssistantSession,
    ),
}

BASELINE_REGISTRY: dict[str, BaselineSpec] = {
    "goal_progression": BaselineSpec(
        constructor=GoalProgressionBaseline,
        prompt_config_key=None,
        session_constructor=GoalProgressionSession,
    ),
    "base": BaselineSpec(constructor=BaseBaseline, prompt_config_key=None),
    "prompt_base": BaselineSpec(
        constructor=PromptBaseBaseline,
        prompt_config_key="prompt_base",
        artifact_loader=SystemPrompt,
    ),
    "interactcomp_react": BaselineSpec(
        constructor=InteractCompReActBaseline,
        prompt_config_key="interactcomp_react",
        artifact_loader=SystemPrompt,
        session_constructor=InteractCompReActSession,
    ),
    "trace2skill": BaselineSpec(
        constructor=Trace2SkillBaseline,
        prompt_config_key="trace2skill",
        artifact_loader=StaticSkill,
    ),
}


@dataclass
class AssistantComponents:
    session: AssistantSession
    baseline: AssistantBaseline
    model_profile: ModelProfile
    prompt: PromptArtifact | None
    action_guard_prompt: SystemPrompt | None
    action_guard_generation: GenerationSettings | None
    action_guard_model: dict[str, Any] | None
    uses_openrouter: bool
    memory_framework: str | None
    baseline_source: Literal["assistant_config", "model_profile"]
    skill_framework: str | None


def configured_components_require_openrouter(config: AssistantConfig) -> bool:
    _resolve_assistant_spec(config)
    profile = load_model_profile(config.models["assistant"])
    resolved_baseline = resolve_active_baseline(config, profile=profile)
    _validate_profile_compatibility(profile, resolved_baseline)
    return profile.provider == "openrouter"


def resolve_active_baseline(
    config: AssistantConfig,
    *,
    profile: ModelProfile | None = None,
) -> ResolvedBaseline:
    """Resolve the effective baseline, including an optional profile-bound skill."""

    resolved_profile = profile or load_model_profile(config.models["assistant"])
    configured_name = config.components.baseline
    configured_spec = _resolve_baseline_spec(config)
    if (configured_name == "goal_progression") != (resolved_profile.goal_progression is not None):
        raise ConfigurationError(
            "goal_progression baseline and model settings must be selected together"
        )
    if config.components.profile_skill_enabled and resolved_profile.skill is not None:
        if configured_name not in {"base", "trace2skill"}:
            raise ConfigurationError(
                "a profile-bound Trace2Skill cannot be combined with prompted base; "
                "use the corresponding non-Trace2Skill model profile instead"
            )
        return ResolvedBaseline(
            name="trace2skill",
            spec=BASELINE_REGISTRY["trace2skill"],
            artifact_path=str(resolved_profile.skill.path),
            source="model_profile",
        )
    artifact_path = (
        str(getattr(config.prompts, configured_spec.prompt_config_key))
        if configured_spec.prompt_config_key is not None
        else None
    )
    return ResolvedBaseline(
        name=configured_name,
        spec=configured_spec,
        artifact_path=artifact_path,
        source="assistant_config",
    )


def build_assistant_components(
    *,
    config: AssistantConfig,
    environment: EnvironmentSettings | None = None,
    client: ChatLLMClient | None = None,
    action_guard_client: ChatLLMClient | None = None,
    action_guard_model: dict[str, Any] | None = None,
    memory_snapshot: Sequence[MemoryEntry] | None = None,
) -> AssistantComponents:
    assistant_spec = _resolve_assistant_spec(config)
    profile = load_model_profile(config.models["assistant"])
    resolved_baseline = resolve_active_baseline(config, profile=profile)
    _validate_profile_compatibility(profile, resolved_baseline)

    prompt: PromptArtifact | None = None
    action_guard_prompt: SystemPrompt | None = None
    action_guard_generation = None
    if resolved_baseline.artifact_path is not None:
        if resolved_baseline.spec.artifact_loader is None:
            raise ConfigurationError(f"baseline {resolved_baseline.name!r} has no artifact loader")
        prompt = resolved_baseline.spec.artifact_loader.load(resolved_baseline.artifact_path)
        if resolved_baseline.name == "interactcomp_react":
            guard_settings = config.interactcomp_react.action_guard
            action_guard = None
            if guard_settings.enabled:
                if action_guard_client is None:
                    raise ConfigurationError(
                        "interactcomp_react action guard requires a separate client built "
                        "from the unified --simulator-model-profile; disable the guard for "
                        "standalone prompt/schema-only use"
                    )
                action_guard_prompt = SystemPrompt.load(config.prompts.interactcomp_action_guard)
                action_guard_generation = guard_settings.generation
                action_guard = InteractCompActionGuard(
                    action_guard_prompt,
                    action_guard_generation,
                    action_guard_client,
                    confirm_rejections=guard_settings.confirm_rejections,
                    cache_verdicts=guard_settings.cache_verdicts,
                    invalid_output_retries=guard_settings.invalid_output_retries,
                )
            baseline = InteractCompReActBaseline(
                prompt,
                action_guard=action_guard,
                semantic_action_budget_per_turn=(
                    config.interactcomp_react.semantic_action_budget_per_turn
                ),
                format_retries_per_slot=config.interactcomp_react.format_retries_per_slot,
            )
        else:
            baseline = resolved_baseline.spec.constructor(prompt)
    else:
        baseline = resolved_baseline.spec.constructor()

    resolved_client = client
    if resolved_client is None:
        resolved_client = OpenAICompatibleChatClient(profile, environment)

    if profile.goal_progression is not None:
        session = GoalProgressionSession(
            resolved_client,
            profile.generation["assistant"],
            baseline,
            profile=profile,
        )
    elif profile.memory is None:
        session_constructor = (
            resolved_baseline.spec.session_constructor or assistant_spec.constructor
        )
        session = session_constructor(
            resolved_client,
            profile.generation["assistant"],
            baseline,
        )
    else:
        store = JsonMemoryStore(profile.memory)
        retriever = build_retriever(profile.memory.retrieval)
        memory_sessions = {
            "exprag": ExpRAGSession,
            "remem": ReMemSession,
        }
        session = memory_sessions[profile.memory.framework](
            resolved_client,
            profile.generation["assistant"],
            baseline,
            settings=profile.memory,
            store=store,
            retriever=retriever,
            memory_snapshot=memory_snapshot,
        )
    return AssistantComponents(
        session=session,
        baseline=baseline,
        model_profile=profile,
        prompt=prompt,
        action_guard_prompt=action_guard_prompt,
        action_guard_generation=action_guard_generation,
        action_guard_model=(action_guard_model if action_guard_prompt is not None else None),
        uses_openrouter=profile.provider == "openrouter",
        memory_framework=(profile.memory.framework if profile.memory is not None else None),
        baseline_source=resolved_baseline.source,
        skill_framework=(
            profile.skill.framework
            if profile.skill is not None and resolved_baseline.source == "model_profile"
            else None
        ),
    )


def _resolve_assistant_spec(config: AssistantConfig) -> AssistantSpec:
    value = config.components.assistant
    if value not in ASSISTANT_REGISTRY:
        raise ConfigurationError(
            f"unknown assistant component {value!r}; available: {sorted(ASSISTANT_REGISTRY)}"
        )
    return ASSISTANT_REGISTRY[value]


def _resolve_baseline_spec(config: AssistantConfig) -> BaselineSpec:
    value = config.components.baseline
    if value not in BASELINE_REGISTRY:
        raise ConfigurationError(
            f"unknown baseline {value!r}; available: {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[value]


def _validate_profile_compatibility(
    profile: ModelProfile,
    resolved_baseline: ResolvedBaseline,
) -> None:
    if resolved_baseline.name == "trace2skill" and profile.memory is not None:
        raise ConfigurationError(
            "the trace2skill baseline is static and requires a memory-free model profile"
        )
    if resolved_baseline.name == "interactcomp_react":
        if profile.memory is not None:
            raise ConfigurationError(
                "the interactcomp_react baseline requires a memory-free model profile"
            )
        if profile.skill is not None:
            raise ConfigurationError(
                "the interactcomp_react baseline cannot be combined with a static skill"
            )
