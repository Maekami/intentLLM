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
)
from assistant.factory import (
    configured_components_require_openrouter as assistant_requires_openrouter,
)
from assistant.prompt import SystemPrompt
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

    @property
    def has_api_key(self) -> bool:
        """Backward-compatible credential readiness check."""

        return not self.missing_credentials

    @property
    def missing_credentials(self) -> list[str]:
        missing: list[str] = []
        if (
            simulator_requires_openrouter(self.simulator_config)
            and not self.simulator_environment.openrouter_api_key
        ):
            missing.append("OPENROUTER_API_KEY (simulator)")
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
        active_prompt = None
        if self.assistant_config.components.baseline == "prompt_base":
            active_prompt = _assistant_prompt_snapshot(
                SystemPrompt.load(self.assistant_config.prompts.prompt_base)
            )

        return {
            "run_overview": {
                "difficulty": self.config.run.difficulty,
                "baseline": self.assistant_config.components.baseline,
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
                },
            },
            "schema_version": 1,
            "batch": {
                "batch_id": batch_id,
                "created_at": created_at,
                "concurrency": concurrency,
                "sample_retries": self.config.run.sample_retries,
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
                "active_baseline": self.assistant_config.components.baseline,
                "active_prompt": active_prompt,
            },
        }

    def build(
        self,
        sample: Sample,
        *,
        output_dir: str | Path,
        seed: int,
        run_id: str | None = None,
    ) -> BuiltInteraction:
        simulator_components = build_simulator_components(
            config=self.simulator_config,
            environment=self.simulator_environment,
        )
        assistant_components = build_assistant_components(
            config=self.assistant_config,
            environment=self.assistant_environment,
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
                "assistant_prompt": (
                    {
                        "name": assistant_components.prompt.name,
                        "path": assistant_components.prompt.path,
                        "hash": assistant_components.prompt.hash,
                    }
                    if assistant_components.prompt is not None
                    else None
                ),
                "sample_id": sample.sample_id,
                "difficulty": self.config.run.difficulty,
                "seed": seed,
            },
        )
        # Keep both simulator-compatible JSONL files present even if the first
        # model call fails before any visible message is produced.
        (audit.run_dir / "events.jsonl").touch(exist_ok=True)
        (audit.run_dir / "transcript.jsonl").touch(exist_ok=True)
        episode = Episode(
            sample=sample,
            difficulty=Difficulty(self.config.run.difficulty),
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
    max_turns: int | None = None,
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
    with _working_directory(assistant_root):
        assistant_config = load_assistant_config(
            assistant_custom,
            cli_overrides=assistant_overrides,
        )
    assistant_config = _absolutize_assistant_config(assistant_config, assistant_root)

    return PreparedPipeline(
        config=config,
        simulator_config=simulator_config,
        assistant_config=assistant_config,
        dataset=DatasetLoader(simulator_config.dataset_path),
        simulator_environment=SimulatorEnvironmentSettings(),
        assistant_environment=AssistantEnvironmentSettings(),
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
        update={"prompt_base": str(_project_file(root, config.prompts.prompt_base))}
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


def _assistant_prompt_snapshot(prompt: SystemPrompt) -> dict[str, Any]:
    return {
        "name": prompt.name,
        "path": prompt.path,
        "prompt_hash": prompt.hash,
        "system": prompt.system,
    }
