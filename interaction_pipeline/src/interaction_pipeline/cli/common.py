from pathlib import Path
from typing import Any

from interaction_pipeline.config import PipelineConfig, load_pipeline_config, resolve_pipeline_path
from interaction_pipeline.prepare import PreparedPipeline, prepare_pipeline


def configure(
    *,
    pipeline_config: Path | None,
    simulator_config: Path | None,
    assistant_config: Path | None,
    simulator_model_profile: str | None,
    assistant_model_profile: str | None,
    baseline: str | None,
    difficulty: str | None,
    seed: int | None,
    concurrency: int | None,
    sample_retries: int | None,
    output_dir: Path | None,
    max_turns: int | None,
) -> tuple[PipelineConfig, PreparedPipeline, Path]:
    overrides: dict[str, Any] = {}
    for key, value in {
        "difficulty": difficulty,
        "seed": seed,
        "concurrency": concurrency,
        "sample_retries": sample_retries,
        "output_dir": str(output_dir.resolve()) if output_dir else None,
    }.items():
        if value is not None:
            overrides.setdefault("run", {})[key] = value
    config = load_pipeline_config(pipeline_config, overrides=overrides)
    prepared = prepare_pipeline(
        config,
        simulator_config_path=simulator_config,
        assistant_config_path=assistant_config,
        simulator_model_profile=simulator_model_profile,
        assistant_model_profile=assistant_model_profile,
        assistant_baseline=baseline,
        max_turns=max_turns,
    )
    resolved_output = resolve_pipeline_path(config.run.output_dir)
    return config, prepared, resolved_output
