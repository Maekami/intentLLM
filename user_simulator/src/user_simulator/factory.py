from dataclasses import dataclass

from user_simulator.config import (
    EnvironmentSettings,
    ModelProfile,
    SimulatorConfig,
    load_model_profile,
)
from user_simulator.controller.base import Controller
from user_simulator.controller.llm_controller import LLMController
from user_simulator.exceptions import ConfigurationError
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.mock_components import (
    MockController,
    MockSatisfactionUpdater,
    MockUserRealizer,
)
from user_simulator.policy.realization import DifficultyRealizationPolicy, RealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy, NodeSelectionPolicy
from user_simulator.realizer.base import UserRealizer
from user_simulator.realizer.llm_realizer import LLMUserRealizer
from user_simulator.satisfaction.base import SatisfactionUpdater
from user_simulator.satisfaction.llm_updater import LLMSatisfactionUpdater


@dataclass(frozen=True)
class ComponentSpec:
    constructor: type
    requires_model: bool


CONTROLLER_REGISTRY: dict[str, ComponentSpec] = {
    "llm_controller": ComponentSpec(
        constructor=LLMController,
        requires_model=True,
    ),
    "mock_controller": ComponentSpec(
        constructor=MockController,
        requires_model=False,
    ),
}
SATISFACTION_REGISTRY: dict[str, ComponentSpec] = {
    "llm_satisfaction_updater": ComponentSpec(
        constructor=LLMSatisfactionUpdater,
        requires_model=True,
    ),
    "mock_satisfaction_updater": ComponentSpec(
        constructor=MockSatisfactionUpdater,
        requires_model=False,
    ),
}
REALIZER_REGISTRY: dict[str, ComponentSpec] = {
    "llm_user_realizer": ComponentSpec(
        constructor=LLMUserRealizer,
        requires_model=True,
    ),
    "mock_user_realizer": ComponentSpec(
        constructor=MockUserRealizer,
        requires_model=False,
    ),
}
SELECTION_REGISTRY = {
    "difficulty_selection": DifficultySelectionPolicy,
}
REALIZATION_REGISTRY = {
    "difficulty_realization": DifficultyRealizationPolicy,
}


@dataclass
class SimulatorComponents:
    controller: Controller
    satisfaction_updater: SatisfactionUpdater
    selection_policy: NodeSelectionPolicy
    realization_policy: RealizationPolicy
    user_realizer: UserRealizer
    prompts: dict[str, PromptTemplate]
    model_profiles: dict[str, ModelProfile]
    uses_openrouter: bool


def configured_components_require_openrouter(config: SimulatorConfig) -> bool:
    controller, satisfaction, realizer = _resolve_component_specs(config)
    required_keys: list[str] = []
    if controller.requires_model:
        required_keys.append("controller")
    if satisfaction.requires_model:
        required_keys.append("satisfaction")
    if realizer.requires_model:
        required_keys.extend(("realizer_clear", "realizer_abstract"))
    missing_profiles = set(required_keys) - set(config.models)
    if missing_profiles:
        raise ConfigurationError(
            f"missing model profiles for components: {sorted(missing_profiles)}"
        )
    return any(
        load_model_profile(config.models[key]).provider == "openrouter"
        for key in required_keys
    )


def build_simulator_components(
    *,
    config: SimulatorConfig,
    environment: EnvironmentSettings | None = None,
) -> SimulatorComponents:
    controller_spec, satisfaction_spec, realizer_spec = _resolve_component_specs(config)
    _validate_name("selection policy", config.components.selection_policy, SELECTION_REGISTRY)
    _validate_name(
        "realization policy",
        config.components.realization_policy,
        REALIZATION_REGISTRY,
    )

    prompts: dict[str, PromptTemplate] = {}
    profiles: dict[str, ModelProfile] = {}
    clients: dict[str, OpenRouterStructuredClient] = {}
    required_resources: dict[str, tuple[str, str]] = {}
    if controller_spec.requires_model:
        required_resources["controller"] = (
            config.prompts.controller,
            "controller_result",
        )
    if satisfaction_spec.requires_model:
        required_resources["satisfaction"] = (
            config.prompts.satisfaction,
            "satisfaction_update_result",
        )
    if realizer_spec.requires_model:
        required_resources.update(
            {
                "realizer_clear": (
                    config.prompts.realizer_clear,
                    "user_generation_result",
                ),
                "realizer_abstract": (
                    config.prompts.realizer_abstract,
                    "user_generation_result",
                ),
            }
        )

    missing_profiles = set(required_resources) - set(config.models)
    if missing_profiles:
        raise ConfigurationError(
            f"missing model profiles for components: {sorted(missing_profiles)}"
        )
    for key, (prompt_path, expected_schema) in required_resources.items():
        prompt = PromptTemplate.load(prompt_path)
        _validate_prompt_schema(key, prompt, expected_schema)
        prompts[key] = prompt
        profiles[key] = load_model_profile(config.models[key])

    uses_openrouter = any(profile.provider == "openrouter" for profile in profiles.values())
    if profiles:
        env = environment or EnvironmentSettings()
        for key, profile in profiles.items():
            clients[key] = OpenRouterStructuredClient(profile, env)

    if controller_spec.requires_model:
        profile = profiles["controller"]
        controller: Controller = controller_spec.constructor(
            clients["controller"],
            profile.generation["controller"],
            prompt=prompts["controller"],
            model_profile_name=profile.profile_name,
            semantic_attempts=profile.retry.max_attempts,
        )
    else:
        controller = controller_spec.constructor()

    if satisfaction_spec.requires_model:
        profile = profiles["satisfaction"]
        satisfaction: SatisfactionUpdater = satisfaction_spec.constructor(
            clients["satisfaction"],
            profile.generation["satisfaction"],
            prompt=prompts["satisfaction"],
            model_profile_name=profile.profile_name,
            semantic_attempts=profile.retry.max_attempts,
        )
    else:
        satisfaction = satisfaction_spec.constructor()

    if realizer_spec.requires_model:
        clear_profile = profiles["realizer_clear"]
        abstract_profile = profiles["realizer_abstract"]
        realizer: UserRealizer = realizer_spec.constructor(
            clients["realizer_clear"],
            clear_profile.generation["realizer_clear"],
            abstract_profile.generation["realizer_abstract"],
            clear_prompt=prompts["realizer_clear"],
            abstract_prompt=prompts["realizer_abstract"],
            model_profile_name=clear_profile.profile_name,
            abstract_client=clients["realizer_abstract"],
            abstract_model_profile_name=abstract_profile.profile_name,
            semantic_attempts=clear_profile.retry.max_attempts,
            abstract_semantic_attempts=abstract_profile.retry.max_attempts,
        )
    else:
        realizer = realizer_spec.constructor()

    selection_class = SELECTION_REGISTRY[config.components.selection_policy]
    realization_class = REALIZATION_REGISTRY[config.components.realization_policy]
    if selection_class is DifficultySelectionPolicy:
        selection_policy = selection_class(
            medium_min_nodes=config.policy.medium_min_nodes,
            medium_max_nodes=config.policy.medium_max_nodes,
        )
    else:
        selection_policy = selection_class()
    return SimulatorComponents(
        controller=controller,
        satisfaction_updater=satisfaction,
        selection_policy=selection_policy,
        realization_policy=realization_class(config.policy.medium_clear_probability),
        user_realizer=realizer,
        prompts=prompts,
        model_profiles=profiles,
        uses_openrouter=uses_openrouter,
    )


def _resolve_component_specs(
    config: SimulatorConfig,
) -> tuple[ComponentSpec, ComponentSpec, ComponentSpec]:
    controller = _resolve_spec(
        "controller",
        config.components.controller,
        CONTROLLER_REGISTRY,
    )
    satisfaction = _resolve_spec(
        "satisfaction updater",
        config.components.satisfaction_updater,
        SATISFACTION_REGISTRY,
    )
    realizer = _resolve_spec(
        "user realizer",
        config.components.user_realizer,
        REALIZER_REGISTRY,
    )
    return controller, satisfaction, realizer


def _resolve_spec(
    label: str,
    value: str,
    registry: dict[str, ComponentSpec],
) -> ComponentSpec:
    _validate_name(label, value, registry)
    return registry[value]


def _validate_name(label: str, value: str, registry: dict[str, object]) -> None:
    if value not in registry:
        raise ConfigurationError(
            f"unknown {label} component {value!r}; available: {sorted(registry)}"
        )


def _validate_prompt_schema(
    label: str,
    prompt: PromptTemplate,
    expected_schema: str,
) -> None:
    if prompt.schema_name != expected_schema:
        raise ConfigurationError(
            f"{label} prompt uses {prompt.schema_name!r}; expected {expected_schema!r}"
        )
