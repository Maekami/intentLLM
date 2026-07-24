from dataclasses import dataclass
from typing import Any

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
from user_simulator.policy.realization import (
    DifficultyRealizationPolicy,
    RealizationPolicy,
)
from user_simulator.policy.selection import DifficultySelectionPolicy, NodeSelectionPolicy
from user_simulator.realizer.base import UserRealizer
from user_simulator.realizer.llm_realizer import LLMUserRealizer
from user_simulator.satisfaction.base import SatisfactionUpdater
from user_simulator.satisfaction.llm_updater import LLMSatisfactionUpdater

CONTROLLER_REGISTRY = {
    "llm_controller": LLMController,
    "mock_controller": MockController,
}
SATISFACTION_REGISTRY = {
    "llm_satisfaction_updater": LLMSatisfactionUpdater,
    "mock_satisfaction_updater": MockSatisfactionUpdater,
}
REALIZER_REGISTRY = {
    "llm_user_realizer": LLMUserRealizer,
    "mock_user_realizer": MockUserRealizer,
}
SELECTION_REGISTRY = {
    "difficulty_selection_v1": DifficultySelectionPolicy,
}
REALIZATION_REGISTRY = {
    "difficulty_realization_v1": DifficultyRealizationPolicy,
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


def build_simulator_components(
    *,
    config: SimulatorConfig,
    environment: EnvironmentSettings | None = None,
    audit: Any | None = None,
) -> SimulatorComponents:
    del audit  # Reserved for component-level audit injection.
    _validate_name("controller", config.components.controller, CONTROLLER_REGISTRY)
    _validate_name(
        "satisfaction updater",
        config.components.satisfaction_updater,
        SATISFACTION_REGISTRY,
    )
    _validate_name("user realizer", config.components.user_realizer, REALIZER_REGISTRY)
    _validate_name("selection policy", config.components.selection_policy, SELECTION_REGISTRY)
    _validate_name(
        "realization policy",
        config.components.realization_policy,
        REALIZATION_REGISTRY,
    )

    prompts = {
        "controller": PromptTemplate.load(config.prompts.controller),
        "satisfaction": PromptTemplate.load(config.prompts.satisfaction),
        "realizer_clear": PromptTemplate.load(config.prompts.realizer_clear),
        "realizer_abstract": PromptTemplate.load(config.prompts.realizer_abstract),
    }
    _validate_prompt_schema("controller", prompts["controller"], "controller_result")
    _validate_prompt_schema("satisfaction", prompts["satisfaction"], "satisfaction_update_result")
    _validate_prompt_schema("realizer_clear", prompts["realizer_clear"], "user_generation_result")
    _validate_prompt_schema(
        "realizer_abstract", prompts["realizer_abstract"], "user_generation_result"
    )

    needs_llm = {
        "controller": config.components.controller == "llm_controller",
        "satisfaction": (config.components.satisfaction_updater == "llm_satisfaction_updater"),
        "realizer_clear": config.components.user_realizer == "llm_user_realizer",
        "realizer_abstract": config.components.user_realizer == "llm_user_realizer",
    }
    profiles = {key: load_model_profile(value) for key, value in config.models.items()}
    required_profiles = {key for key, required in needs_llm.items() if required}
    missing_profiles = required_profiles - set(profiles)
    if missing_profiles:
        raise ConfigurationError(
            f"missing model profiles for components: {sorted(missing_profiles)}"
        )
    clients = {}
    if required_profiles:
        env = environment or EnvironmentSettings()
        for key in required_profiles:
            clients[key] = OpenRouterStructuredClient(profiles[key], env)

    if needs_llm["controller"]:
        profile = profiles["controller"]
        controller: Controller = LLMController(
            clients["controller"],
            profile.generation["controller"],
            prompt=prompts["controller"],
            model_profile_name=profile.profile_name,
            semantic_attempts=profile.retry.max_attempts,
        )
    else:
        controller = MockController()

    if needs_llm["satisfaction"]:
        profile = profiles["satisfaction"]
        satisfaction: SatisfactionUpdater = LLMSatisfactionUpdater(
            clients["satisfaction"],
            profile.generation["satisfaction"],
            prompt=prompts["satisfaction"],
            model_profile_name=profile.profile_name,
            semantic_attempts=profile.retry.max_attempts,
        )
    else:
        satisfaction = MockSatisfactionUpdater()

    if needs_llm["realizer_clear"]:
        clear_profile = profiles["realizer_clear"]
        abstract_profile = profiles["realizer_abstract"]
        realizer: UserRealizer = LLMUserRealizer(
            clients["realizer_clear"],
            clear_profile.generation["realizer_clear"],
            abstract_profile.generation["realizer_abstract"],
            clear_prompt=prompts["realizer_clear"],
            abstract_prompt=prompts["realizer_abstract"],
            model_profile_name=clear_profile.profile_name,
            abstract_client=clients["realizer_abstract"],
            abstract_model_profile_name=abstract_profile.profile_name,
        )
    else:
        realizer = MockUserRealizer()

    selection_class = SELECTION_REGISTRY[config.components.selection_policy]
    realization_class = REALIZATION_REGISTRY[config.components.realization_policy]
    return SimulatorComponents(
        controller=controller,
        satisfaction_updater=satisfaction,
        selection_policy=selection_class(),
        realization_policy=realization_class(config.policy.medium_clear_probability),
        user_realizer=realizer,
        prompts=prompts,
        model_profiles=profiles,
    )


def _validate_name(label: str, value: str, registry: dict[str, type]) -> None:
    if value not in registry:
        raise ConfigurationError(
            f"unknown {label} component {value!r}; available: {sorted(registry)}"
        )


def _validate_prompt_schema(label: str, prompt: PromptTemplate, expected_schema: str) -> None:
    if prompt.schema_name != expected_schema:
        raise ConfigurationError(
            f"{label} prompt uses {prompt.schema_name!r}; expected {expected_schema!r}"
        )
