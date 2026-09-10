from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DifficultyName = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class ConversationMessage:
    turn_index: int
    role: Literal["user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeTrace:
    run_dir: Path
    run_id: str
    sample_id: str
    difficulty: DifficultyName
    assistant_model: str
    intent_node_ids: tuple[str, ...]
    initial_exposed_node_ids: tuple[str, ...]
    events: tuple[dict[str, Any], ...]
    conversation: tuple[ConversationMessage, ...]
    source_outcome: Literal["completed", "budget_exhausted"]

    @property
    def episode_id(self) -> str:
        return f"{self.sample_id}:{self.run_id}"


@dataclass(frozen=True)
class DagTurnMetrics:
    all_node_exposure_turn: int
    all_node_satisfaction_turn: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenMetric:
    assistant_tokens: int
    fallback_turns: tuple[int, ...] = ()


@dataclass(frozen=True)
class AITRMetric:
    raw_score: float
    normalized_score: float
    score: float
    reason: str
    cached: bool = False


@dataclass(frozen=True)
class EpisodeMetricRecord:
    episode_id: str
    sample_id: str
    run_dir: str
    difficulty: DifficultyName
    assistant_model: str
    source_outcome: str
    all_node_exposure_turn: int
    all_node_satisfaction_turn: int
    assistant_tokens: int | None
    aitr_raw_score: float | None = None
    aitr_normalized_score: float | None = None
    aitr: float | None = None
    aitr_reason: str | None = None
    aitr_cached: bool = False
    aitr_error: str | None = None
    aitr_judge_config_hash: str | None = None
    aitr_prompt_version: str | None = None
    aitr_prompt_hash: str | None = None
    warnings: tuple[str, ...] = ()
    token_error: str | None = None

    @property
    def status(self) -> str:
        if self.assistant_tokens is None:
            return "tokens_incomplete"
        if self.aitr_error:
            return "aitr_error"
        if self.aitr is None:
            return "aitr_skipped"
        return "complete"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class EvaluationFailure:
    run_dir: str
    error_type: str
    message: str
    sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    records: list[EpisodeMetricRecord] = field(default_factory=list)
    failures: list[EvaluationFailure] = field(default_factory=list)
