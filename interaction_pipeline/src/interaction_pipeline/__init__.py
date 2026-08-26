"""Automated simulator/assistant interaction pipeline."""

from interaction_pipeline.config import PipelineConfig, load_pipeline_config
from interaction_pipeline.core import RunResult, run_interaction
from interaction_pipeline.prepare import PreparedPipeline, prepare_pipeline

__all__ = [
    "PipelineConfig",
    "PreparedPipeline",
    "RunResult",
    "load_pipeline_config",
    "prepare_pipeline",
    "run_interaction",
]
