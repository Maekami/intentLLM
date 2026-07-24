from pydantic import BaseModel, ConfigDict, Field

from user_simulator.domain.enums import RealizationMode, SatisfactionLevel


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateExposureDecision(StrictModel):
    node_id: str = Field(description="Candidate intent node ID")
    exposable: bool = Field(description="Whether conversation supports exposing this node")
    reason: str = Field(description="Short audit explanation")


class ControllerResult(StrictModel):
    decisions: list[CandidateExposureDecision]
    end_reachable: bool = Field(description="Structural END reachability, not termination")
    summary: str = Field(description="Concise audit summary")


class NormalizedControllerResult(StrictModel):
    decisions: list[CandidateExposureDecision]
    newly_exposed: list[str]
    end_reachable: bool
    violations: list[str] = Field(default_factory=list)
    summary: str


class NodeSatisfactionDecision(StrictModel):
    node_id: str = Field(description="Exposed intent node ID")
    status: SatisfactionLevel
    reason: str = Field(description="Short audit explanation")


class SatisfactionUpdateResult(StrictModel):
    updates: list[NodeSatisfactionDecision]
    summary: str = Field(description="Concise audit summary")


class UserGenerationResult(StrictModel):
    user_message: str = Field(min_length=1)
    selected_node_ids: list[str]
    realization_mode: RealizationMode
    coverage_check: dict[str, bool]
    contains_unsupported_intent: bool
    summary: str = Field(description="Concise audit summary")


class SelectionResult(StrictModel):
    unresolved_queue: list[str]
    selected_nodes: list[str]


class TurnResult(StrictModel):
    terminal: bool
    user_message: str | None = None
    state: "EpisodeState"
    audit: dict = Field(default_factory=dict)


from user_simulator.domain.state import EpisodeState
