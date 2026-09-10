from __future__ import annotations

import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from assistant.config import (
    AssistantConfig,
)
from assistant.config import (
    EnvironmentSettings as AssistantEnvironmentSettings,
)
from assistant.config import (
    load_config as load_assistant_config,
)
from assistant.config import (
    load_model_profile as load_assistant_model_profile,
)
from assistant.factory import (
    build_assistant_components,
    resolve_active_baseline,
)
from assistant.factory import (
    configured_components_require_openrouter as assistant_requires_openrouter,
)
from assistant.goal_progression.prompts import snapshot as gp_snapshot
from assistant.memory.models import MemoryEntry
from assistant.prompt import SystemPrompt
from assistant.skill import StaticSkill
from user_simulator.audit.logger import AuditLogger, read_git_commit
from user_simulator.config import (
    EnvironmentSettings as SimulatorEnvironmentSettings,
)
from user_simulator.config import (
    SimulatorConfig,
)
from user_simulator.config import (
    load_config as load_simulator_config,
)
from user_simulator.config import (
    load_model_profile as load_simulator_model_profile,
)
from user_simulator.data.loader import DatasetLoader
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.factory import (
    build_simulator_components,
)
from user_simulator.factory import (
    configured_components_require_openrouter as simulator_requires_openrouter,
)
from user_simulator.llm.prompt import PromptTemplate

from interaction_pipeline.config import PipelineConfig, resolve_pipeline_path
from interaction_pipeline.interactcomp_guard import (
    SimulatorProfileGuardClient,
    action_guard_model_metadata,
)


@dataclass
class BuiltInteraction:
    episode: Episode
    assistant: Any
    audit: AuditLogger


@dataclass
class PreparedPipeline:
    config: PipelineConfig
    simulator_config: SimulatorConfig
    assistant_config: AssistantConfig
    dataset: DatasetLoader
    simulator_environment: SimulatorEnvironmentSettings
    assistant_environment: AssistantEnvironmentSettings
    action_guard_model_profile_path: str | None = None
    action_guard_model_source: str | None = None

    @property
    def has_api_key(self) -> bool:
        """Backward-compatible credential readiness check."""

        return not self.missing_credentials

    @property
    def missing_credentials(self) -> list[str]:
        missing: list[str] = []
        simulator_key_missing = (
            simulator_requires_openrouter(self.simulator_config)
            and not self.simulator_environment.openrouter_api_key
        )
        if simulator_key_missing:
            missing.append("OPENROUTER_API_KEY (simulator)")
        if (
            self.action_guard_model_profile_path is not None
            and load_simulator_model_profile(self.action_guard_model_profile_path).provider
            == "openrouter"
            and not self.simulator_environment.openrouter_api_key
            and not simulator_key_missing
        ):
            missing.append("OPENROUTER_API_KEY (action guard)")
        if (
            assistant_requires_openrouter(self.assistant_config)
            and not self.assistant_environment.openrouter_api_key
        ):
            missing.append("OPENROUTER_API_KEY (assistant)")
        return missing

    def batch_config_snapshot(
        self,
        *,
        batch_id: str,
        created_at: str,
        samples: list[Sample],
        concurrency: int,
        update_memory: bool = True,
        retry_policy: str = "all_failures",
    ) -> dict[str, Any]:
        """Return the resolved, secret-free configuration for one batch run."""

        simulator_profiles = {
            key: {
                "source_path": path,
                "settings": load_simulator_model_profile(path).model_dump(mode="json"),
            }
            for key, path in self.simulator_config.models.items()
        }
        simulator_prompts = {
            key: _simulator_prompt_snapshot(PromptTemplate.load(path))
            for key, path in {
                "controller": self.simulator_config.prompts.controller,
                "satisfaction": self.simulator_config.prompts.satisfaction,
                "realizer_clear": self.simulator_config.prompts.realizer_clear,
                "realizer_abstract": self.simulator_config.prompts.realizer_abstract,
            }.items()
        }
        assistant_profile_path = self.assistant_config.models["assistant"]
        assistant_profile = load_assistant_model_profile(assistant_profile_path)
        assistant_memory_store = None
        if assistant_profile.memory is not None:
            memory_path = assistant_profile.memory.path
            if not memory_path.is_file():
                raise ValueError(f"assistant memory store does not exist: {memory_path}")
            memory_stat = memory_path.stat()
            assistant_memory_store = {
                "path": str(memory_path),
                "sha256": _sha256_file(memory_path),
                "size_bytes": memory_stat.st_size,
                "modified_at_ns": memory_stat.st_mtime_ns,
            }
        resolved_baseline = resolve_active_baseline(
            self.assistant_config,
            profile=assistant_profile,
        )
        active_prompt = None
        if resolved_baseline.name in {"prompt_base", "interactcomp_react"}:
            active_prompt = _assistant_prompt_snapshot(
                SystemPrompt.load(resolved_baseline.artifact_path or "")
            )
        elif resolved_baseline.name == "trace2skill":
            active_prompt = _assistant_prompt_snapshot(
                StaticSkill.load(resolved_baseline.artifact_path or "")
            )
        action_guard_enabled = (
            resolved_baseline.name == "interactcomp_react"
            and self.assistant_config.interactcomp_react.action_guard.enabled
        )
        action_guard_prompt = (
            SystemPrompt.load(self.assistant_config.prompts.interactcomp_action_guard)
            if action_guard_enabled
            else None
        )
        guard_profile = None
        guard_model = None
        if action_guard_enabled:
            if (
                self.action_guard_model_profile_path is None
                or self.action_guard_model_source is None
            ):
                raise ValueError(
                    "enabled InteractComp action guard has no resolved unified simulator profile"
                )
            guard_profile = load_simulator_model_profile(self.action_guard_model_profile_path)
            guard_model = {
                **action_guard_model_metadata(
                    profile=guard_profile,
                    source=self.action_guard_model_source,
                    source_path=self.action_guard_model_profile_path,
                ),
                "settings": guard_profile.model_dump(mode="json"),
            }

        return {
            "run_overview": {
                "difficulty": self.config.run.difficulty,
                "baseline": resolved_baseline.name,
                "models": {
                    "simulator": {
                        key: {
                            "profile_name": value["settings"]["profile_name"],
                            "model_id": value["settings"]["model_id"],
                        }
                        for key, value in simulator_profiles.items()
                    },
                    "assistant": {
                        "profile_name": assistant_profile.profile_name,
                        "model_id": assistant_profile.model_id,
                    },
                    **(
                        {
                            "action_guard": {
                                "profile_name": guard_profile.profile_name,
                                "model_id": guard_profile.model_id,
                                "source": self.action_guard_model_source,
                            }
                        }
                        if guard_profile is not None
                        else {}
                    ),
                },
            },
            "schema_version": 2,
            "batch": {
                "batch_id": batch_id,
                "created_at": created_at,
                "concurrency": concurrency,
                "sample_retries": self.config.run.sample_retries,
                "retry_policy": retry_policy,
                "update_memory": update_memory,
                "sample_count": len(samples),
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "seed": self.config.run.seed + index,
                    }
                    for index, sample in enumerate(samples)
                ],
            },
            "git_commit": read_git_commit(),
            "dataset": {
                "path": self.simulator_config.dataset_path,
                "sha256": _sha256_file(self.simulator_config.dataset_path),
                "selected_sample_count": len(samples),
                "selected_order": "source_order_or_explicit_cli_order",
            },
            "pipeline": self.config.model_dump(mode="json"),
            "simulator": {
                "config": self.simulator_config.model_dump(mode="json"),
                "model_profiles": simulator_profiles,
                "prompts": simulator_prompts,
            },
            "assistant": {
                "config": self.assistant_config.model_dump(mode="json"),
                "model_profile": {
                    "source_path": assistant_profile_path,
                    "settings": assistant_profile.model_dump(mode="json"),
                },
                "memory_store": assistant_memory_store,
                "configured_baseline": self.assistant_config.components.baseline,
                "active_baseline": resolved_baseline.name,
                "baseline_source": resolved_baseline.source,
                "active_prompt": active_prompt,
                "goal_progression": gp_snapshot(assistant_profile),
                "action_guard": {
                    "enabled": action_guard_enabled,
                    "prompt": (
                        _assistant_prompt_snapshot(action_guard_prompt)
                        if action_guard_prompt is not None
                        else None
                    ),
                    "generation": (
                        self.assistant_config.interactcomp_react.action_guard.generation.model_dump(
                            mode="json"
                        )
                        if action_guard_enabled
                        else None
                    ),
                    "model_profile": guard_model,
                    "confirm_rejections": (
                        self.assistant_config.interactcomp_react.action_guard.confirm_rejections
                        if action_guard_enabled
                        else None
                    ),
                    "cache_verdicts": (
                        self.assistant_config.interactcomp_react.action_guard.cache_verdicts
                        if action_guard_enabled
                        else None
                    ),
                    "invalid_output_retries": (
                        self.assistant_config.interactcomp_react.action_guard.invalid_output_retries
                        if action_guard_enabled
                        else None
                    ),
                },
            },
        }

    def build(
        self,
        sample: Sample,
        *,
        output_dir: str | Path,
        seed: int,
        run_id: str | None = None,
        assistant_memory_snapshot: tuple[MemoryEntry, ...] | None = None,
        execution_context: dict[str, Any] | None = None,
        difficulty: Difficulty | str | None = None,
    ) -> BuiltInteraction:
        effective_difficulty = (
            difficulty
            if isinstance(difficulty, Difficulty)
            else Difficulty(difficulty or self.config.run.difficulty)
        )
        simulator_components = build_simulator_components(
            config=self.simulator_config,
            environment=self.simulator_environment,
        )
        action_guard_client = None
        action_guard_model = None
        active_assistant_profile = load_assistant_model_profile(
            self.assistant_config.models["assistant"]
        )
        active_baseline = resolve_active_baseline(
            self.assistant_config,
            profile=active_assistant_profile,
        )
        if (
            active_baseline.name == "interactcomp_react"
            and self.assistant_config.interactcomp_react.action_guard.enabled
        ):
            if (
                self.action_guard_model_profile_path is None
                or self.action_guard_model_source is None
            ):
                raise ValueError(
                    "enabled InteractComp action guard has no resolved unified simulator profile"
                )
            guard_profile = load_simulator_model_profile(self.action_guard_model_profile_path)
            action_guard_client = SimulatorProfileGuardClient(
                guard_profile,
                self.simulator_environment,
            )
            action_guard_model = action_guard_model_metadata(
                profile=guard_profile,
                source=self.action_guard_model_source,
                source_path=self.action_guard_model_profile_path,
            )
        assistant_components = build_assistant_components(
            config=self.assistant_config,
            environment=self.assistant_environment,
            action_guard_client=action_guard_client,
            action_guard_model=action_guard_model,
            memory_snapshot=assistant_memory_snapshot,
        )
        resolved_run_id = run_id or f"{_safe_slug(sample.sample_id)}_{uuid.uuid4().hex[:10]}"
        audit = AuditLogger(
            sample.sample_id,
            output_dir=output_dir,
            run_id=resolved_run_id,
            level=self.config.run.audit_level,
            enabled=True,
            config_snapshot={
                "pipeline": self.config.model_dump(mode="json"),
                "simulator": self.simulator_config.model_dump(mode="json"),
                "simulator_model_profiles": {
                    key: profile.model_dump(mode="json")
                    for key, profile in simulator_components.model_profiles.items()
                },
                "assistant": self.assistant_config.model_dump(mode="json"),
                "assistant_model_profile": assistant_components.model_profile.model_dump(
                    mode="json"
                ),
                "assistant_goal_progression": gp_snapshot(assistant_components.model_profile),
                "assistant_prompt": (
                    {
                        "name": assistant_components.prompt.name,
                        "path": assistant_components.prompt.path,
                        "hash": assistant_components.prompt.hash,
                    }
                    if assistant_components.prompt is not None
                    else None
                ),
                "assistant_action_guard": {
                    "enabled": assistant_components.action_guard_prompt is not None,
                    "prompt": (
                        {
                            "name": assistant_components.action_guard_prompt.name,
                            "path": assistant_components.action_guard_prompt.path,
                            "hash": assistant_components.action_guard_prompt.hash,
                        }
                        if assistant_components.action_guard_prompt is not None
                        else None
                    ),
                    "generation": (
                        assistant_components.action_guard_generation.model_dump(mode="json")
                        if assistant_components.action_guard_generation is not None
                        else None
                    ),
                    "model_profile": assistant_components.action_guard_model,
                    "confirm_rejections": (
                        self.assistant_config.interactcomp_react.action_guard.confirm_rejections
                        if assistant_components.action_guard_prompt is not None
                        else None
                    ),
                    "cache_verdicts": (
                        self.assistant_config.interactcomp_react.action_guard.cache_verdicts
                        if assistant_components.action_guard_prompt is not None
                        else None
                    ),
                    "invalid_output_retries": (
                        self.assistant_config.interactcomp_react.action_guard.invalid_output_retries
                        if assistant_components.action_guard_prompt is not None
                        else None
                    ),
                },
                "assistant_baseline": {
                    "configured": self.assistant_config.components.baseline,
                    "active": assistant_components.baseline.name,
                    "source": assistant_components.baseline_source,
                    "skill_framework": assistant_components.skill_framework,
                },
                "sample_id": sample.sample_id,
                "difficulty": effective_difficulty.value,
                "seed": seed,
                "execution_context": execution_context,
            },
        )
        # Keep both simulator-compatible JSONL files present even if the first
        # model call fails before any visible message is produced.
        (audit.run_dir / "events.jsonl").touch(exist_ok=True)
        (audit.run_dir / "transcript.jsonl").touch(exist_ok=True)
        episode = Episode(
            sample=sample,
            difficulty=effective_difficulty,
            seed=seed,
            controller=simulator_components.controller,
            satisfaction_updater=simulator_components.satisfaction_updater,
            selection_policy=simulator_components.selection_policy,
            realization_policy=simulator_components.realization_policy,
            user_realizer=simulator_components.user_realizer,
            audit_logger=audit,
            max_turns=self.simulator_config.policy.max_turns,
            monotonic_satisfaction=self.simulator_config.policy.monotonic_satisfaction,
            auto_expose_backbone_on_empty_queue=(
                self.simulator_config.policy.auto_expose_backbone_on_empty_queue
            ),
        )
        return BuiltInteraction(
            episode=episode,
            assistant=assistant_components.session,
            audit=audit,
        )


def prepare_pipeline(
    config: PipelineConfig,
    *,
    simulator_config_path: str | Path | None = None,
    assistant_config_path: str | Path | None = None,
    simulator_model_profile: str | None = None,
    assistant_model_profile: str | None = None,
    assistant_baseline: str | None = None,
    assistant_trace2skill_path: str | Path | None = None,
    assistant_profile_skill_enabled: bool | None = None,
    react_action_guard: bool | None = None,
    max_turns: int | None = None,
    simulator_dataset_path: str | Path | None = None,
) -> PreparedPipeline:
    simulator_root = resolve_pipeline_path(config.projects.simulator_root)
    assistant_root = resolve_pipeline_path(config.projects.assistant_root)
    simulator_custom = (
        Path(simulator_config_path).expanduser().resolve()
        if simulator_config_path is not None
        else _custom_config_path(simulator_root, config.projects.simulator_config)
    )
    assistant_custom = (
        Path(assistant_config_path).expanduser().resolve()
        if assistant_config_path is not None
        else _custom_config_path(assistant_root, config.projects.assistant_config)
    )
    simulator_profile_was_explicit = simulator_model_profile is not None
    simulator_model_profile = _resolve_explicit_profile(simulator_model_profile)
    assistant_model_profile = _resolve_explicit_profile(assistant_model_profile)

    simulator_overrides: dict[str, Any] = {}
    if simulator_model_profile:
        simulator_overrides["models"] = {
            key: simulator_model_profile
            for key in (
                "controller",
                "satisfaction",
                "realizer_clear",
                "realizer_abstract",
            )
        }
    if max_turns is not None:
        simulator_overrides.setdefault("policy", {})["max_turns"] = max_turns
    if simulator_dataset_path is not None:
        simulator_overrides["dataset_path"] = str(simulator_dataset_path)
    with _working_directory(simulator_root):
        simulator_config = load_simulator_config(
            simulator_custom,
            cli_overrides=simulator_overrides,
        )
    simulator_config = _absolutize_simulator_config(simulator_config, simulator_root)

    assistant_overrides: dict[str, Any] = {}
    if assistant_model_profile:
        assistant_overrides.setdefault("models", {})["assistant"] = assistant_model_profile
    if assistant_baseline:
        assistant_overrides.setdefault("components", {})["baseline"] = assistant_baseline
    if assistant_profile_skill_enabled is not None:
        assistant_overrides.setdefault("components", {})["profile_skill_enabled"] = (
            assistant_profile_skill_enabled
        )
    if react_action_guard is not None:
        react_overrides = assistant_overrides.setdefault("interactcomp_react", {})
        react_overrides.setdefault("action_guard", {})["enabled"] = react_action_guard
    if assistant_trace2skill_path is not None:
        assistant_overrides.setdefault("prompts", {})["trace2skill"] = str(
            assistant_trace2skill_path
        )
    with _working_directory(assistant_root):
        assistant_config = load_assistant_config(
            assistant_custom,
            cli_overrides=assistant_overrides,
        )
    assistant_config = _absolutize_assistant_config(assistant_config, assistant_root)
    assistant_profile = load_assistant_model_profile(assistant_config.models["assistant"])
    if (
        assistant_profile.skill is not None
        and assistant_config.components.profile_skill_enabled
        and assistant_trace2skill_path is not None
    ):
        raise ValueError(
            "--trace2skill-skill cannot override a skill bound by --assistant-model-profile"
        )
    resolved_baseline = resolve_active_baseline(assistant_config, profile=assistant_profile)
    if resolved_baseline.name == "goal_progression":
        gp_snapshot(assistant_profile)  # Validate enabled prompt files before any model call.
    if resolved_baseline.name in {"prompt_base", "interactcomp_react"}:
        SystemPrompt.load(resolved_baseline.artifact_path or "")
    elif resolved_baseline.name == "trace2skill":
        StaticSkill.load(resolved_baseline.artifact_path or "")
    if (
        resolved_baseline.name == "interactcomp_react"
        and assistant_config.interactcomp_react.action_guard.enabled
    ):
        SystemPrompt.load(assistant_config.prompts.interactcomp_action_guard)

    action_guard_model_profile_path = None
    action_guard_model_source = None
    if (
        resolved_baseline.name == "interactcomp_react"
        and assistant_config.interactcomp_react.action_guard.enabled
    ):
        action_guard_model_profile_path = _uniform_simulator_model_profile_path(simulator_config)
        action_guard_model_source = (
            "cli.simulator_model_profile"
            if simulator_profile_was_explicit
            else "simulator.models.unified"
        )

    return PreparedPipeline(
        config=config,
        simulator_config=simulator_config,
        assistant_config=assistant_config,
        dataset=DatasetLoader(simulator_config.dataset_path),
        simulator_environment=SimulatorEnvironmentSettings(),
        assistant_environment=AssistantEnvironmentSettings(),
        action_guard_model_profile_path=action_guard_model_profile_path,
        action_guard_model_source=action_guard_model_source,
    )


def _absolutize_simulator_config(config: SimulatorConfig, root: Path) -> SimulatorConfig:
    prompts = config.prompts.model_copy(
        update={
            key: str(_project_file(root, getattr(config.prompts, key)))
            for key in ("controller", "satisfaction", "realizer_clear", "realizer_abstract")
        }
    )
    models = {key: str(_model_profile_file(root, value)) for key, value in config.models.items()}
    return config.model_copy(
        update={
            "dataset_path": str(_project_file(root, config.dataset_path)),
            "prompts": prompts,
            "models": models,
        }
    )


def _absolutize_assistant_config(config: AssistantConfig, root: Path) -> AssistantConfig:
    prompts = config.prompts.model_copy(
        update={
            "prompt_base": str(_project_file(root, config.prompts.prompt_base)),
            "interactcomp_react": str(_project_file(root, config.prompts.interactcomp_react)),
            "interactcomp_action_guard": str(
                _project_file(root, config.prompts.interactcomp_action_guard)
            ),
            "trace2skill": str(_project_file(root, config.prompts.trace2skill)),
        }
    )
    models = {key: str(_model_profile_file(root, value)) for key, value in config.models.items()}
    return config.model_copy(update={"prompts": prompts, "models": models})


def _custom_config_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return _project_file(root, value)


def _project_file(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _model_profile_file(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    direct = (root / candidate).resolve()
    if direct.exists():
        return direct
    return (root / "configs/models" / f"{value}.yaml").resolve()


def _uniform_simulator_model_profile_path(config: SimulatorConfig) -> str:
    keys = ("controller", "satisfaction", "realizer_clear", "realizer_abstract")
    missing = [key for key in keys if key not in config.models]
    if missing:
        raise ValueError(
            "enabled InteractComp action guard requires a unified simulator model; "
            f"missing simulator model entries: {missing}"
        )
    paths = {str(Path(config.models[key]).expanduser().resolve()) for key in keys}
    if len(paths) != 1:
        raise ValueError(
            "enabled InteractComp action guard requires one unified simulator model YAML; "
            "pass --simulator-model-profile explicitly instead of selecting a clear or "
            "abstract realizer profile"
        )
    path = paths.pop()
    # Validate the exact source now so configuration errors happen before a
    # batch directory or any paid model call is created.
    load_simulator_model_profile(path)
    return path


def _resolve_explicit_profile(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return value


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _safe_slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (result or "sample")[:100]


def _simulator_prompt_snapshot(prompt: PromptTemplate) -> dict[str, Any]:
    source = Path(prompt.path).read_bytes()
    return {
        "name": prompt.name,
        "schema_name": prompt.schema_name,
        "path": prompt.path,
        "source_sha256": sha256(source).hexdigest(),
        "system": prompt.system,
        "user_template": prompt.user_template,
    }


def _assistant_prompt_snapshot(prompt: SystemPrompt | StaticSkill) -> dict[str, Any]:
    return {
        "name": prompt.name,
        "path": prompt.path,
        "prompt_hash": prompt.hash,
        "system": prompt.system,
    }


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
