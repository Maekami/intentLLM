"""Synthetic state-shape and latest-message/repair boundary regressions."""

import pytest
from gp_helpers import Client, profile, ref, semantic, session
from test_gp_identity_v2 import constraint

from assistant.goal_progression.contracts import output_schema, validate_wire_required
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.events import EvidenceStore, IdentityIndex
from assistant.goal_progression.schemas import SemanticSnapshot
from assistant.goal_progression.tracker import normalize_snapshot

jsonschema = pytest.importorskip("jsonschema")


@pytest.mark.parametrize(
    "case,valid",
    [
        ("base", True),
        ("current_anchor", False),
        ("adjacent_no_anchor", False),
        ("addressed_context", True),
        ("active_no_work", False),
        ("orphan_constraint", False),
        ("available_without_answer", False),
        ("available_with_answer", True),
    ],
)
def test_state_wire_and_local_contract_agree(case, valid):
    raw = semantic(needs=True)
    if case == "current_anchor":
        raw["goals"][0]["anchor_goal_ids"] = ["new:adjacent"]
    elif case == "adjacent_no_anchor":
        raw["goals"][1]["anchor_goal_ids"] = []
    elif case == "addressed_context":
        raw["goals"][1].update(status="addressed", remaining_work="", anchor_goal_ids=[])
    elif case == "active_no_work":
        raw["goals"][0]["remaining_work"] = ""
    elif case == "orphan_constraint":
        raw["constraints"] = [constraint()]
        raw["constraints"][0]["applies_to"] = []
    elif case.startswith("available_"):
        raw["needs"][0]["status"] = "available"
        raw["needs"][0]["answer_refs"] = [ref()] if case == "available_with_answer" else []
    registry = IdentityIndex()
    wire = output_schema(SemanticSnapshot, registry=registry)
    jsonschema.Draft202012Validator.check_schema(wire)
    assert jsonschema.Draft202012Validator(wire).is_valid(raw) is valid
    value = SemanticSnapshot.model_validate(raw)
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make a synthetic card."}])
    if valid:
        normalize_snapshot(value, registry, store, "t1", profile().goal_progression.policy)
    else:
        with pytest.raises(ContractError):
            normalize_snapshot(value, registry, store, "t1", profile().goal_progression.policy)


async def test_tracker_receives_latest_request_separately_from_previous_snapshot():
    c = Client()
    s = session(client=c)
    await s.respond("Make a synthetic card.")
    await s.respond("Before proceeding, explain what you need from me now.")
    call = [call for call in c.calls if call["role"] == "tracker"][-1]
    context = call["context"]
    assert context["latest_user"] == {
        "event_id": "m3",
        "role": "user",
        "content": "Before proceeding, explain what you need from me now.",
    }
    assert context["previous_snapshot"]["goals"]
    assert context["visible_history"][-1]["content"] == context["latest_user"]["content"]


async def test_orphan_constraint_repair_names_exact_field_and_original_local_refs():
    bad = semantic(adjacent=False)
    bad["constraints"] = [constraint()]
    bad["constraints"][0]["applies_to"] = []

    def repair(call):
        error = call["repair"]["errors"][0]
        assert error["field_path"] == "$.constraints[0].applies_to"
        assert "NONEMPTY" in error["detail"] and "new:current" in error["detail"]
        fixed = semantic(adjacent=False)
        fixed["constraints"] = [constraint()]
        return fixed

    s = session(client=Client({"tracker": [bad, repair]}))
    await s.respond("Make a synthetic card.")
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "tracker:contract.reference": 1
    }


def test_reverse_constraint_index_is_runtime_owned_and_not_wire_input():
    raw = semantic()
    raw["constraints"] = [constraint(gid="new:adjacent")]
    wire = output_schema(SemanticSnapshot, registry=IdentityIndex())
    assert jsonschema.Draft202012Validator(wire).is_valid(raw)
    validate_wire_required(raw, wire, "tracker")
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make a matte card."}])
    projected, _, _ = normalize_snapshot(
        SemanticSnapshot.model_validate(raw),
        IdentityIndex(),
        store,
        "t1",
        profile().goal_progression.policy,
    )
    assert projected.goals[0].constraint_ids == []
    assert projected.goals[1].constraint_ids == [projected.constraints[0].ref]
    raw["goals"][0]["constraint_ids"] = ["new:finish"]
    assert not jsonschema.Draft202012Validator(wire).is_valid(raw)
    with pytest.raises(ContractError) as error:
        validate_wire_required(raw, wire, "tracker")
    assert error.value.field_path == "$.goals[0].constraint_ids"


async def test_constraint_scope_change_rebuilds_inverse_without_stale_copy():
    first = semantic()
    first["constraints"] = [constraint()]

    def move(call):
        value = call["context"]["previous_snapshot"]
        assert all("constraint_ids" not in g for g in value["goals"])
        value["constraints"][0]["applies_to"] = [value["goals"][1]["ref"]]
        value["constraints"][0]["evidence_refs"] = [ref("m3")]
        return value

    s = session(client=Client({"tracker": [first, move]}))
    await s.respond("Use a matte finish for the card only.")
    assert s.state.goals[0].constraint_ids == ["c0001"]
    await s.respond("Apply that finish only to the storage label now.")
    assert s.state.goals[0].constraint_ids == []
    assert s.state.goals[1].constraint_ids == ["c0001"]
    assert not s.last_call_metadata["goal_progression"]["retry_counts"]
