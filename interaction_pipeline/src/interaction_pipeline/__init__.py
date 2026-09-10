"""Automated simulator/assistant interaction pipeline."""

from interaction_pipeline.config import PipelineConfig, load_pipeline_config
from interaction_pipeline.core import RunResult, run_interaction
from interaction_pipeline.evolution import run_memory_evolution
from interaction_pipeline.evolution_config import EvolutionConfig, load_evolution_config
from interaction_pipeline.prepare import PreparedPipeline, prepare_pipeline
from interaction_pipeline.trace2skill import (
    Trace2SkillCollectionResult,
    Trace2SkillCorpus,
    Trace2SkillEvolutionResult,
    load_trace2skill_corpus,
    run_trace2skill_collection,
)
from interaction_pipeline.trace2skill_build import run_trace2skill_build
from interaction_pipeline.trace2skill_config import (
    Trace2SkillConfig,
    load_trace2skill_config,
)

__all__ = [
    "EvolutionConfig",
    "PipelineConfig",
    "PreparedPipeline",
    "RunResult",
    "Trace2SkillCollectionResult",
    "Trace2SkillConfig",
    "Trace2SkillCorpus",
    "Trace2SkillEvolutionResult",
    "load_evolution_config",
    "load_pipeline_config",
    "load_trace2skill_config",
    "load_trace2skill_corpus",
    "prepare_pipeline",
    "run_interaction",
    "run_memory_evolution",
    "run_trace2skill_build",
    "run_trace2skill_collection",
]
