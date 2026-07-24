from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DagNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    node_id: str
    node_type: Literal["intent", "terminal"]
    user_turn_index: int | None = None
    reason_label: str | None = None
    reason_text: str | None = None
    surface_user_message: str | None = None
    node_intent: str = Field(min_length=1)


class DagEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    edge_id: str
    source: str
    target: str
    reason: str | None = None


class ReasonDAG(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[DagNode]
    edges: list[DagEdge]


class Sample(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: str = Field(min_length=1)
    reason_dag: ReasonDAG
    retained_original_fields: dict[str, Any] = Field(default_factory=dict)

    @property
    def task_summary(self) -> str:
        return str(self.retained_original_fields.get("task_summary", ""))

    @property
    def task_expectation(self) -> str:
        return str(self.retained_original_fields.get("task_expectation", ""))
