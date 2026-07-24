from pydantic import BaseModel, ConfigDict, Field

from user_simulator.llm.schemas import (
    CandidateExposureDecisionV2,
    ControllerResultV2,
    NodeCoverageDecisionV2,
    NodeSatisfactionDecisionV2,
    SatisfactionUpdateResultV2,
    UserGenerationResultV2,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CandidateExposureDecision = CandidateExposureDecisionV2
ControllerResult = ControllerResultV2
NodeSatisfactionDecision = NodeSatisfactionDecisionV2
SatisfactionUpdateResult = SatisfactionUpdateResultV2
NodeCoverageDecision = NodeCoverageDecisionV2
UserGenerationResult = UserGenerationResultV2


class NormalizedControllerResult(StrictModel):
    decisions: list[CandidateExposureDecision]
    newly_exposed: list[str]
    end_reachable: bool
    violations: list[str] = Field(default_factory=list)
    summary: str


class SelectionResult(StrictModel):
    unresolved_queue: list[str]
    selected_nodes: list[str]


class TurnResult(StrictModel):
    terminal: bool
    user_message: str | None = None
    state: "EpisodeState"
    audit: dict = Field(default_factory=dict)


from user_simulator.domain.state import EpisodeState
