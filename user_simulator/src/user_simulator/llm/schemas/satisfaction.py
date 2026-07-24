from pydantic import Field

from user_simulator.domain.enums import SatisfactionLevel
from user_simulator.llm.schemas.base import StrictWireModel


class NodeSatisfactionDecisionV2(StrictWireModel):
    node_id: str = Field(min_length=1, description="Exposed intent node ID")
    status: SatisfactionLevel
    reason: str = Field(
        min_length=1,
        max_length=300,
        description="Assistant evidence and the principal remaining gap, if any",
    )


class SatisfactionUpdateResultV2(StrictWireModel):
    updates: list[NodeSatisfactionDecisionV2]
    summary: str = Field(
        min_length=1, max_length=500, description="Concise satisfaction audit summary"
    )
