"""Configurable OpenRouter and local vLLM assistant baselines."""

from assistant.config import (
    AssistantConfig,
    EnvironmentSettings,
    GoalProgressionSettings,
    MemorySettings,
    ModelProfile,
    SkillSettings,
    load_config,
    load_model_profile,
)
from assistant.factory import (
    AssistantComponents,
    ResolvedBaseline,
    build_assistant_components,
    resolve_active_baseline,
)
from assistant.goal_progression import GoalProgressionBaseline, GoalProgressionSession
from assistant.memory.sessions import ExpRAGSession, ReMemSession
from assistant.session import AssistantSession
from assistant.skill import StaticSkill

__all__ = [
    "AssistantComponents",
    "AssistantConfig",
    "AssistantSession",
    "EnvironmentSettings",
    "ExpRAGSession",
    "GoalProgressionBaseline",
    "GoalProgressionSession",
    "GoalProgressionSettings",
    "MemorySettings",
    "ModelProfile",
    "ReMemSession",
    "ResolvedBaseline",
    "SkillSettings",
    "StaticSkill",
    "build_assistant_components",
    "load_config",
    "load_model_profile",
    "resolve_active_baseline",
]
