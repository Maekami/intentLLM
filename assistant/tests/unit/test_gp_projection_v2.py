"""Explicit current/adjacent wire views preserve the canonical state and ownership."""

import copy

import pytest
from gp_helpers import Client, profile, ref, semantic, session

from assistant.goal_progression.contracts import output_schema, validate_wire_required
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.events import EvidenceStore, IdentityIndex
from assistant.goal_progression.schemas import SemanticProjection
from assistant.goal_progression.tracker import normalize_projection

jsonschema = pytest.importorskip("jsonschema")


def projection():
    data = semantic(needs=True)
    goals = data.pop("goals")
    return {**data, "current_goals": goals[:1], "adjacent_goals": goals[1:]}


@pytest.mark.parametrize(
    "case,valid",
    [
        ("both", True),
        ("no_adjacent", True),
        ("no_current", False),
        ("swapped_scope", False),
        ("wrong_new_id", False),
        ("invented_changed_id", False),
    ],
)
def test_projection_wire_preserves_scope_and_typed_identity_boundaries(case, valid):
    data = projection()
    if case == "no_adjacent":
        data["adjacent_goals"] = []
    elif case == "no_current":
        data["current_goals"] = []
    elif case == "swapped_scope":
        data["current_goals"], data["adjacent_goals"] = (
            data["adjacent_goals"],
            data["current_goals"],
        )
    elif case == "wrong_new_id":
        data["adjacent_goals"][0]["ref"] = "g0999"
    elif case == "invented_changed_id":
        data["changed_goal_ids"] = ["g0999"]
    registry = IdentityIndex()
    schema = output_schema(SemanticProjection, registry=registry)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert jsonschema.Draft202012Validator(schema).is_valid(data) is valid
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make a card."}])
    value = SemanticProjection.model_validate(data)
    if valid:
        snapshot, _, _ = normalize_projection(
            value,
            registry,
            store,
            "t1",
            profile().goal_progression.policy,
        )
        assert snapshot.goals[0].scope == "current"
    else:
        with pytest.raises(ContractError):
            normalize_projection(value, registry, store, "t1", profile().goal_progression.policy)


def test_projection_reports_adjacent_field_path_without_changing_its_scope():
    data = projection()
    data["adjacent_goals"][0].update(basis="explicit", evidence_refs=[ref("m2")])
    original = copy.deepcopy(data)
    store = EvidenceStore(
        [
            {"event_id": "m1", "role": "user", "content": "Make a card."},
            {"event_id": "m2", "role": "assistant", "content": "Draft card."},
        ]
    )
    with pytest.raises(ContractError) as error:
        normalize_projection(
            SemanticProjection.model_validate(data),
            IdentityIndex(),
            store,
            "t2",
            profile().goal_progression.policy,
        )
    assert error.value.field_path == "$.adjacent_goals[0].evidence_refs"
    assert data == original


def test_old_mixed_wire_shape_is_not_silently_accepted():
    schema = output_schema(SemanticProjection, registry=IdentityIndex())
    with pytest.raises(ContractError):
        validate_wire_required(semantic(), schema, "tracker")


async def test_projection_and_canonical_snapshot_are_both_auditable():
    events = []
    s = session(client=Client())
    await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    raw = next(p for k, p in events if k == "assistant_tracker_validated")["structured_result"]
    assert "current_goals" in raw and "adjacent_goals" in raw and "goals" not in raw
    canonical = next(p for k, p in events if k == "assistant_gp_snapshot_projected")["snapshot"]
    assert len(canonical["goals"]) == len(raw["current_goals"]) + len(raw["adjacent_goals"])
    assert "constraint_ids" in canonical["goals"][0]
    assert not s.last_call_metadata["goal_progression"]["retry_counts"]
