from pydantic import Field

from user_simulator.domain.enums import RealizationMode
from user_simulator.llm.schemas.base import StrictWireModel


class NodeCoverageDecisionV2(StrictWireModel):
    node_id: str = Field(min_length=1, description="Selected intent node ID")
    covered: bool = Field(description="Whether the generated message expresses this node")


class UserGenerationResultV2(StrictWireModel):
    user_message: str = Field(min_length=1, max_length=4000, description="One natural user message")
    selected_node_ids: list[str]
    realization_mode: RealizationMode
    coverage: list[NodeCoverageDecisionV2]
    contains_unsupported_intent: bool
    summary: str = Field(
        min_length=1, max_length=500, description="Concise generation audit summary"
    )
