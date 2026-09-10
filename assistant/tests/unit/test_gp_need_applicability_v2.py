"""Need identity/history stays stable when its relevant working goal changes."""

import pytest
from gp_helpers import goal, profile, ref
from test_gp_identity_v2 import registered_snapshot

from assistant.goal_progression.context import ContextBuilder
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.events import EvidenceStore, InteractionIndex
from assistant.goal_progression.schemas import EvidenceRef, Goal
from assistant.goal_progression.tracker import normalize_snapshot


def reassociated(*, available=False):
    snapshot, registry, previous_store = registered_snapshot()
    old_ref = snapshot.needs[0].ref
    old_origin = registry.entries[old_ref].copy()
    store = EvidenceStore(
        [
            *previous_store.events,
            {"event_id": "m2", "role": "assistant", "content": "What edge shape is needed?"},
            {
                "event_id": "m3",
                "role": "user",
                "content": (
                    "The shape is round; use that shape on a matching label too."
                    if available
                    else "Make a matching label too; I will provide the shape later."
                ),
            },
        ]
    )
    snapshot.goals = [Goal.model_validate(goal("new:matching_label", event="m3"))]
    snapshot.constraints = []
    snapshot.needs[0].goal_id = "new:matching_label"
    if available:
        snapshot.needs[0].status = "available"
        snapshot.needs[0].answer_refs = [EvidenceRef.model_validate(ref("m3"))]
    value, candidate, _ = normalize_snapshot(
        snapshot, registry, store, "t2", profile().goal_progression.policy
    )
    return value, candidate, store, old_ref, old_origin


def test_reassociation_keeps_original_identity_and_receipt_history():
    value, registry, store, need_id, origin = reassociated()
    interactions = InteractionIndex(
        requests={
            need_id: {
                "issued_count": 1,
                "last_issued_event_id": "m2",
                "last_issued_turn": 1,
            }
        }
    )
    view = interactions.view(value, registry, store)[need_id]
    assert registry.entries[need_id] == origin
    assert view["issued_count"] == 1
    assert view["request_eligible"] is False
    assert view["reason"] == "already_issued"
    assert value.goals[0].ref != origin["scope_id"]


def test_available_information_moves_with_evidence_without_obsolete_goal():
    value, registry, store, need_id, origin = reassociated(available=True)
    builder = ContextBuilder(store, registry, InteractionIndex())
    context = builder.planner(value, "intra")
    assert context["needs"][0]["ref"] == need_id
    assert context["needs"][0]["answer_refs"] == [ref("m3")]
    assert origin["scope_id"] not in {item["ref"] for item in context["goals"]}
    assert context["request_eligibility"][need_id]["request_eligible"] is False
    annotated = registry.annotated_history(store)
    identity = next(
        item
        for event in annotated
        for item in event.get("identity_refs", [])
        if item["id"] == need_id
    )
    assert identity["origin_goal_id"] == origin["scope_id"]


def test_reassociation_does_not_allow_dangling_goal_links():
    value, registry, store, _, _ = reassociated()
    value.needs[0].goal_id = "new:undeclared"
    with pytest.raises(ContractError) as error:
        normalize_snapshot(value, registry, store, "t3", profile().goal_progression.policy)
    assert error.value.field_path == "$.needs[0].goal_id"


def test_reassociation_is_not_automatic_when_tracker_keeps_an_obsolete_reference():
    value, registry, store, _, origin = reassociated()
    value.needs[0].goal_id = origin["scope_id"]
    with pytest.raises(ContractError, match="not declared"):
        normalize_snapshot(value, registry, store, "t3", profile().goal_progression.policy)
