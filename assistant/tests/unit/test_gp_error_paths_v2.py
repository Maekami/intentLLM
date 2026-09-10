"""Semantic identity errors retain strict validation and usable repair locations."""

import copy
import json

import pytest
from gp_helpers import Client, current_plan, need, profile, ref, semantic, session

from assistant.exceptions import InvalidModelResponseError
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.events import EvidenceStore, IdentityIndex
from assistant.goal_progression.repair_context import repair_excerpt
from assistant.goal_progression.schemas import SemanticProjection
from assistant.goal_progression.tracker import normalize_projection


def projection():
    data = semantic(needs=True)
    goals = data.pop("goals")
    return {**data, "current_goals": goals[:1], "adjacent_goals": goals[1:]}


def store():
    return EvidenceStore(
        [
            {"event_id": "m1", "role": "user", "content": "Make a synthetic card."},
            {"event_id": "m2", "role": "assistant", "content": "A possible draft."},
        ]
    )


@pytest.mark.parametrize("case", ["current", "adjacent", "need", "cross_kind"])
def test_duplicate_ids_are_rejected_at_the_wire_field_without_mutation(case):
    data = projection()
    if case == "current":
        data["current_goals"].append(copy.deepcopy(data["current_goals"][0]))
        path, entity = "$.current_goals[1].ref", data["current_goals"][1]
    elif case == "adjacent":
        data["adjacent_goals"][0]["ref"] = data["current_goals"][0]["ref"]
        path, entity = "$.adjacent_goals[0].ref", data["adjacent_goals"][0]
    elif case == "need":
        data["needs"].append(copy.deepcopy(data["needs"][0]))
        path, entity = "$.needs[1].ref", data["needs"][1]
    else:
        data["constraints"] = [
            {
                "ref": "new:shape",
                "key": "shape",
                "value": "curved",
                "applies_to": ["new:current"],
                "condition": None,
                "basis": "explicit",
                "evidence_refs": [ref()],
            }
        ]
        path, entity = "$.needs[0].ref", data["needs"][0]
    original = copy.deepcopy(data)
    registry = IdentityIndex()
    with pytest.raises(ContractError) as exc:
        normalize_projection(
            SemanticProjection.model_validate(data),
            registry,
            store(),
            "t1",
            profile().goal_progression.policy,
        )
    error = exc.value
    assert error.code == "contract.reference" and error.owner == "tracker"
    assert error.field_path == path
    assert "globally unique" in error.detail and entity["ref"] in error.detail
    excerpt = json.loads(repair_excerpt(json.dumps(data), {"errors": [error.payload()]}, 2000))
    assert excerpt["rejected_fragment"] == entity
    assert data == original and registry.entries == {}


@pytest.mark.parametrize(
    "case,path",
    [
        ("origin", "$.needs[0].evidence_refs"),
        ("answer", "$.needs[0].answer_refs"),
        ("unavailable", "$.needs[0].evidence_refs"),
        ("authorization", "$.needs[0].renewed_authorization_ref"),
    ],
)
def test_need_errors_use_fields_not_canonical_ids(case, path):
    data = projection()
    item = data["needs"][0]
    if case == "origin":
        item["evidence_refs"] = []
    elif case == "answer":
        item["status"] = "available"
    elif case == "unavailable":
        item.update(status="unavailable", evidence_refs=[ref("m2")])
    else:
        item["renewed_authorization_ref"] = ref("m2")
    with pytest.raises(ContractError) as exc:
        normalize_projection(
            SemanticProjection.model_validate(data),
            IdentityIndex(),
            store(),
            "t1",
            profile().goal_progression.policy,
        )
    assert exc.value.field_path == path


async def test_duplicate_identity_retry_receives_the_conflicting_entity():
    raw = semantic(needs=True)
    raw["needs"].append(need())

    def fix(call):
        payload = json.loads(call["messages"][1]["content"])
        assert call["repair"]["errors"][0]["field_path"] == "$.needs[1].ref"
        excerpt = json.loads(payload["previous_output_excerpt"])
        assert excerpt["rejected_fragment"] == raw["needs"][1]
        repaired = copy.deepcopy(raw)
        repaired["needs"][1].update(ref="new:storage_size", description="Storage size")
        return repaired

    c = Client({"tracker": [raw, fix]})
    s = session(client=c)
    await s.respond("Make a synthetic card.")
    assert len({n.ref for n in s.state.needs}) == 2
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "tracker:contract.reference": 1,
    }


@pytest.mark.parametrize("has_partial", [True, False])
async def test_client_truncation_preserves_received_output_for_audit_and_bounded_repair(
    has_partial,
):
    partial = '{"items":[{"target":"' + "REPEATED SYNTHETIC TEXT " * 300
    error = InvalidModelResponseError("truncated")
    attempt = {"finish_reason": "length", "input_tokens": 12, "output_tokens": 2048}
    if has_partial:
        attempt["raw_output"] = partial
    error.call_metadata = {"finish_reason": "length", "transport_attempts": [attempt]}

    def fix(call):
        payload = json.loads(call["messages"][1]["content"])
        assert "Do not continue" in call["repair"]["instruction"]
        if has_partial:
            assert payload["previous_output_excerpt"] == partial[:2000]
        else:
            assert "previous_output_excerpt" not in payload
        return current_plan(call["context"])

    c = Client({"intra": [error, fix]})
    s = session(client=c)
    events = []
    await s.respond("Make a synthetic card.", audit_sink=lambda k, p: events.append((k, p)))
    failed = next(p for k, p in events if k == "assistant_intra_failed")
    assert failed["response_received"] is True
    if has_partial:
        assert failed["raw_output"] == partial
        received = [p for k, p in events if k == "assistant_intra_raw_result"]
        assert received[0]["raw_output"] == partial
        assert received[0]["response_received"] is True
    else:
        assert "raw_output" not in failed
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "intra:output.truncated": 1,
    }
    assert [x["generation"]["max_completion_tokens"] for x in c.calls if x["role"] == "intra"] == [
        2048,
        3072,
    ]

