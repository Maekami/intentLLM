from pydantic import Field

from user_simulator.llm.schemas.base import StrictWireModel


class CandidateExposureDecisionV2(StrictWireModel):
    node_id: str = Field(min_length=1, description="Candidate intent node ID")
    exposable: bool = Field(description="Whether evidence permits exposing this node")
    reason: str = Field(
        min_length=1, max_length=300, description="Concise evidence-based justification"
    )


class ControllerResultV2(StrictWireModel):
    decisions: list[CandidateExposureDecisionV2]
    end_reachable: bool = Field(description="Structural END permission, not termination")
    summary: str = Field(
        min_length=1, max_length=500, description="Concise controller audit summary"
    )
