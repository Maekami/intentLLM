"""Declared-state changes belong to runtime, not model-authored update identities."""

import copy

import pytest
from gp_helpers import Client, semantic, session
from test_gp_identity_v2 import constraint

from assistant.goal_progression.contracts import output_schema, validate_wire_required
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.schemas import EvidenceRef, SemanticSnapshot
from assistant.goal_progression.state_delta import updated_goal_ids


def state():
    data = semantic(needs=True)
    data["constraints"] = [constraint()]
    return SemanticSnapshot.model_validate(data)


def test_state_delta_is_scoped_to_goal_and_need_changes_not_metadata():
    before = state()
    assert updated_goal_ids(before, None) == ["new:current", "new:adjacent"]
    after = before.model_copy(deep=True)
    before.changed_goal_ids = ["new:current"]
    assert updated_goal_ids(after, before) == []
    after.needs[0].status = "available"
    after.needs[0].answer_refs = [EvidenceRef(event_id="m3")]
    assert updated_goal_ids(after, before) == ["new:current"]


def test_constraint_change_updates_all_applicable_goals_without_modifying_state():
    before = state()
    before.constraints[0].applies_to = [g.ref for g in before.goals]
    after = before.model_copy(deep=True)
    after.constraints[0].value = "glossy"
    original = copy.deepcopy(after.model_dump())
    assert updated_goal_ids(after, before) == ["new:current", "new:adjacent"]
    assert after.model_dump() == original


def test_retired_goals_are_not_emitted_as_live_update_references():
    before = state()
    after = before.model_copy(deep=True)
    after.goals = after.goals[:1]
    assert updated_goal_ids(after, before) == []


def test_wire_rejects_model_authored_change_metadata():
    data = semantic()
    data["changed_goal_ids"] = ["new:status_update"]
    with pytest.raises(ContractError) as error:
        validate_wire_required(data, output_schema(SemanticSnapshot), "tracker")
    assert error.value.field_path == "$.changed_goal_ids"


async def test_runtime_projects_delta_and_does_not_broadcast_stale_change_flags():
    c = Client()
    s = session(client=c)
    await s.respond("Make a card.")
    assert set(s.state.changed_goal_ids) == {g.ref for g in s.state.goals}
    await s.respond("No change to the requirements.")
    assert (
        "changed_goal_ids"
        not in [x for x in c.calls if x["role"] == "tracker"][-1]["context"]["previous_snapshot"]
    )
    assert s.state.changed_goal_ids == []
    assert not s.last_call_metadata["goal_progression"]["retry_counts"]
