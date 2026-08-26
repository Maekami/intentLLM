from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from user_simulator.domain.enums import RealizationMode, SatisfactionLevel


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class CandidateExposureDecision(StrictModel):
    node_id: str = Field(min_length=1, description="Candidate intent node ID")
    exposable: bool = Field(description="Whether evidence permits exposing this node")
    reason: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise evidence-based justification",
    )


class ControllerResult(StrictModel):
    decisions: list[CandidateExposureDecision]
    summary: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise controller audit summary",
    )


class NodeSatisfactionDecision(StrictModel):
    node_id: str = Field(min_length=1, description="Exposed intent node ID")
    status: SatisfactionLevel
    reason: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise assistant evidence and justification for the status",
    )
    remaining_gap: str | None = Field(
        min_length=1,
        max_length=1024,
        description=(
            "Missing deliverable explicitly required by the partially satisfied node; "
            "visible user facts may specialize it but cannot add objectives, and it "
            "must exclude assistant-introduced prerequisites, optional personalization, "
            "later-node information, and new task facts; null for all other statuses"
        ),
    )

    @model_validator(mode="after")
    def validate_remaining_gap(self) -> NodeSatisfactionDecision:
        if self.status == SatisfactionLevel.PARTIALLY_SATISFIED:
            if self.remaining_gap is None or not self.remaining_gap.strip():
                raise ValueError("partially_satisfied requires a non-empty remaining_gap")
        elif self.remaining_gap is not None:
            raise ValueError("remaining_gap must be null unless status is partially_satisfied")
        return self


class UnsatisfiedNodeSatisfactionDecision(NodeSatisfactionDecision):
    status: Literal[SatisfactionLevel.UNSATISFIED]
    remaining_gap: None = Field(
        description="Must be null because status is unsatisfied",
    )


class PartiallySatisfiedNodeSatisfactionDecision(NodeSatisfactionDecision):
    status: Literal[SatisfactionLevel.PARTIALLY_SATISFIED]
    remaining_gap: str = Field(
        min_length=1,
        max_length=1024,
        description=(
            "Missing deliverable explicitly required by this node; visible user facts "
            "may specialize it but cannot add objectives, and it excludes assistant-"
            "introduced prerequisites, optional personalization, later-node information, "
            "and new task facts"
        ),
    )


class SatisfiedNodeSatisfactionDecision(NodeSatisfactionDecision):
    status: Literal[SatisfactionLevel.SATISFIED]
    remaining_gap: None = Field(
        description="Must be null because status is satisfied",
    )


StructuredNodeSatisfactionDecision = (
    UnsatisfiedNodeSatisfactionDecision
    | PartiallySatisfiedNodeSatisfactionDecision
    | SatisfiedNodeSatisfactionDecision
)


def make_node_satisfaction_decision(
    *,
    node_id: str,
    status: SatisfactionLevel,
    reason: str,
    remaining_gap: str | None,
) -> StructuredNodeSatisfactionDecision:
    decision_types = {
        SatisfactionLevel.UNSATISFIED: UnsatisfiedNodeSatisfactionDecision,
        SatisfactionLevel.PARTIALLY_SATISFIED: PartiallySatisfiedNodeSatisfactionDecision,
        SatisfactionLevel.SATISFIED: SatisfiedNodeSatisfactionDecision,
    }
    return decision_types[status](
        node_id=node_id,
        status=status,
        reason=reason,
        remaining_gap=remaining_gap,
    )


class SatisfactionUpdateResult(StrictModel):
    updates: list[StructuredNodeSatisfactionDecision]
    summary: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise satisfaction audit summary",
    )


class NodeCoverageDecision(StrictModel):
    node_id: str = Field(min_length=1, description="Selected intent node ID")
    covered: bool = Field(description="Model-reported coverage check for this selected node")


class UserGenerationResult(StrictModel):
    user_message: str = Field(
        min_length=1,
        max_length=4000,
        description="One natural user message",
    )
    selected_node_ids: list[str] = Field(
        min_length=1,
        description="Selected intent node IDs in required order",
    )
    realization_mode: RealizationMode
    coverage: list[NodeCoverageDecision] = Field(
        min_length=1,
        description="One model-reported coverage check per selected intent node",
    )
    contains_unsupported_task_content: bool = Field(
        description=(
            "Model-reported check for task-relevant content not grounded in selected nodes "
            "or facts previously stated by the user"
        )
    )
    summary: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise generation audit summary",
    )


class NormalizedControllerResult(StrictModel):
    decisions: list[CandidateExposureDecision]
    newly_exposed: list[str]
    violations: list[str] = Field(default_factory=list)
    summary: str


class SelectionResult(StrictModel):
    unresolved_queue: list[str]
    selected_nodes: list[str]


class TurnResult(StrictModel):
    terminal: bool
    user_message: str | None = None
    state: EpisodeState
    audit: dict = Field(default_factory=dict)


from user_simulator.domain.state import EpisodeState
