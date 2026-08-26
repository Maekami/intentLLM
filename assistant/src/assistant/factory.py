from dataclasses import dataclass

from assistant.baselines.base import AssistantBaseline, BaseBaseline
from assistant.baselines.prompt_base import PromptBaseBaseline
from assistant.config import AssistantConfig, EnvironmentSettings, ModelProfile, load_model_profile
from assistant.exceptions import ConfigurationError
from assistant.llm.base import ChatLLMClient
from assistant.llm.openrouter_client import OpenAICompatibleChatClient
from assistant.memory.retrieval import build_retriever
from assistant.memory.sessions import ExpRAGSession, ReMemSession
from assistant.memory.store import JsonMemoryStore
from assistant.prompt import SystemPrompt
from assistant.session import AssistantSession


@dataclass(frozen=True)
class AssistantSpec:
    constructor: type[AssistantSession]


@dataclass(frozen=True)
class BaselineSpec:
    constructor: type[AssistantBaseline]
    prompt_config_key: str | None


ASSISTANT_REGISTRY: dict[str, AssistantSpec] = {
    "llm_assistant": AssistantSpec(
        constructor=AssistantSession,
    ),
}

BASELINE_REGISTRY: dict[str, BaselineSpec] = {
    "base": BaselineSpec(constructor=BaseBaseline, prompt_config_key=None),
    "prompt_base": BaselineSpec(
        constructor=PromptBaseBaseline,
        prompt_config_key="prompt_base",
    ),
}


@dataclass
class AssistantComponents:
    session: AssistantSession
    baseline: AssistantBaseline
    model_profile: ModelProfile
    prompt: SystemPrompt | None
    uses_openrouter: bool
    memory_framework: str | None


def configured_components_require_openrouter(config: AssistantConfig) -> bool:
    _resolve_assistant_spec(config)
    return load_model_profile(config.models["assistant"]).provider == "openrouter"


def build_assistant_components(
    *,
    config: AssistantConfig,
    environment: EnvironmentSettings | None = None,
    client: ChatLLMClient | None = None,
) -> AssistantComponents:
    assistant_spec = _resolve_assistant_spec(config)
    baseline_spec = _resolve_baseline_spec(config)
    profile = load_model_profile(config.models["assistant"])

    prompt: SystemPrompt | None = None
    if baseline_spec.prompt_config_key is not None:
        prompt_path = getattr(config.prompts, baseline_spec.prompt_config_key)
        prompt = SystemPrompt.load(prompt_path)
        baseline = baseline_spec.constructor(prompt)
    else:
        baseline = baseline_spec.constructor()

    resolved_client = client
    if resolved_client is None:
        resolved_client = OpenAICompatibleChatClient(profile, environment)

    if profile.memory is None:
        session = assistant_spec.constructor(
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
        )
    return AssistantComponents(
        session=session,
        baseline=baseline,
        model_profile=profile,
        prompt=prompt,
        uses_openrouter=profile.provider == "openrouter",
        memory_framework=(profile.memory.framework if profile.memory is not None else None),
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
