"""GP v2: shallow role-owned contracts, without semantic transition history."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Text = Annotated[str, StringConstraints(min_length=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceRef(ContractModel):
    event_id: Text
    block_id: str | None = None


class Goal(ContractModel):
    ref: Text
    scope: Literal["current", "adjacent"]
    basis: Literal["explicit", "inferred"]
    status: Literal["active", "paused", "addressed"]
    description: Text
    remaining_work: str = Field(
        description="The unmet outcome, not an instruction to ask/wait or an answer draft. "
        "For current goals this is owed now; for adjacent goals this is potential next work."
    )
    anchor_goal_ids: list[str] = Field(
        default_factory=list,
        description="Active adjacent goals cite declared active current goals or addressed results. "
        "Not paused goals or unfulfilled adjacent candidates. Current goals use an empty list."
    )
    constraint_ids: list[str] = Field(default_factory=list)
    material_refs: list[EvidenceRef] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef]


class Constraint(ContractModel):
    ref: Text
    key: Text
    value: Text
    applies_to: list[str]
    condition: str | None = None
    basis: Literal["explicit", "inferred"]
    evidence_refs: list[EvidenceRef]


class InformationNeed(ContractModel):
    ref: Text
    goal_id: Text = Field(
        description="The declared goal this information is relevant to in the current projection. "
        "Its applicability may change while the same information keeps its stable Need ref "
        "and asking history. This is not the immutable origin goal."
    )
    description: Text = Field(
        description="One independently answerable user fact or identifiable source, "
        "not a bundle of unrelated intake questions."
    )
    status: Literal["unknown", "available", "unavailable"]
    answer_refs: list[EvidenceRef] = Field(default_factory=list)
    renewed_authorization_ref: EvidenceRef | None = None
    evidence_refs: list[EvidenceRef]


class SemanticSnapshot(ContractModel):
    goals: list[Goal]
    constraints: list[Constraint]
    needs: list[InformationNeed]
    changed_goal_ids: list[str] = Field(default_factory=list)

    def goals_for(self, scope: str) -> list[Goal]:
        return [g for g in self.goals if g.scope == scope and g.status == "active"]


class SemanticProjection(ContractModel):
    """Tracker wire view; runtime keeps the existing normalized snapshot format."""

    current_goals: list[Goal] = Field(description="Outcomes required by the present request.")
    adjacent_goals: list[Goal] = Field(
        description="Grounded, distinct next outcomes; inferred candidates are allowed."
    )
    constraints: list[Constraint]
    needs: list[InformationNeed]
    changed_goal_ids: list[str] = Field(default_factory=list)


class DeliveryProposal(ContractModel):
    goal_id: Text
    target: Text = Field(
        description="A brief instruction specifying the concrete output and required coverage. "
        "Not a draft reply, introductory sentence, user-facing question or promise. "
        "Generator writes the actual deliverable."
    )
    required_need_ids: list[str] = Field(default_factory=list)
    material_refs: list[EvidenceRef] = Field(default_factory=list)


class RequestProposal(ContractModel):
    need_id: Text
    question_text: Text = Field(
        description="The exact standalone question to display verbatim to the user, asking "
        "for this Need's information. Not instructions to Generator or a question-writing task."
    )


class IntraItem(ContractModel):
    goal_id: Text
    action: Literal["advance", "revise", "clarify"] | None
    deliveries: list[DeliveryProposal]
    request: RequestProposal | None
    blocked_by: list[str] = Field(
        default_factory=list,
        description="Exact unresolved Need IDs only; no explanations or annotated IDs.",
    )


class IntraPlan(ContractModel):
    items: list[IntraItem]


class InterPlan(ContractModel):
    action: Literal["none", "elicit", "anticipate"]
    goal_id: str | None
    deliveries: list[DeliveryProposal]
    request: RequestProposal | None


class JointPlan(ContractModel):
    intra: IntraPlan
    inter: InterPlan


class IntraPacket(ContractModel):
    semantic: SemanticSnapshot
    plan: IntraPlan


class InterPacket(ContractModel):
    semantic: SemanticSnapshot
    plan: InterPlan


class DeliveryUnit(DeliveryProposal):
    unit_id: Text
    source: Literal["intra", "inter", "runtime"]
    criticality: Literal["required", "optional"]
    kind: Literal["delivery", "boundary"] = "delivery"


class SelectedRequest(RequestProposal):
    source: Literal["intra", "inter"]
    criticality: Literal["required", "optional"]


class BlockedCurrent(ContractModel):
    goal_id: Text
    need_ids: list[str]


class TurnContract(ContractModel):
    contract_id: Text
    snapshot_version: int
    deliveries: list[DeliveryUnit]
    requests: list[SelectedRequest]
    blocked_current: list[BlockedCurrent]


class BodyUnit(ContractModel):
    unit_id: Text
    text: Text = Field(
        description="The complete user-facing deliverable, not a preface or promise."
    )


class RealizationIssue(ContractModel):
    code: Literal["context.missing", "realization.unrealizable"]
    unit_id: Text
    input_refs: list[str]
    detail: Text = Field(
        description="A specific missing required input or conflicting instruction."
    )


class GeneratorOutput(ContractModel):
    units: list[BodyUnit]
    issues: list[RealizationIssue]


class CurrentRealization(ContractModel):
    """no_intra only: explicitly combine current policy and realization.

    Current IDs: intra:<goal_id>:<delivery_index>; boundary:<goal_id>.
    Inter IDs are supplied in the work order, never decided by Generator.
    """

    intra: IntraPlan
    units: list[BodyUnit]
    issues: list[RealizationIssue]


def none_inter() -> InterPlan:
    return InterPlan(action="none", goal_id=None, deliveries=[], request=None)
