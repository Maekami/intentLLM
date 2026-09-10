"""Synthetic regressions for useful, bounded feedback without output rewriting."""

import json

import pytest
from gp_helpers import Client, goal, profile, semantic, session
from test_gp_identity_v2 import registered_snapshot

from assistant.goal_progression.contracts import output_schema
from assistant.goal_progression.repair_context import repair_excerpt
from assistant.goal_progression.schemas import SemanticSnapshot
from assistant.goal_progression.tracker import normalize_snapshot


def test_repair_excerpt_preserves_the_rejected_object_beyond_long_unrelated_prefix():
    data = {"history": "UNRELATED " * 1000, "needs": [{"ref": "n0001", "goal_id": "bad"}]}
    raw = json.dumps(data)
    correction = {"errors": [{"field_path": "$.needs[0].goal_id"}]}
    excerpt = repair_excerpt(raw, correction, 200)
    assert len(excerpt) <= 200
    assert json.loads(excerpt)["rejected_fragment"] == data["needs"][0]
    assert "UNRELATED" not in excerpt
    assert json.loads(raw) == data


@pytest.mark.parametrize("raw,path", [("invalid", "$.items"), ('{"x": 1}', "$.missing")])
def test_repair_excerpt_fallback_respects_configured_budget(raw, path):
    assert repair_excerpt(raw, {"errors": [{"field_path": path}]}, 4) == raw[:4]


def test_existing_need_can_apply_to_a_new_declared_goal_without_new_identity():
    jsonschema = pytest.importorskip("jsonschema")
    snapshot, registry, store = registered_snapshot()
    data = snapshot.model_dump()
    data.pop("changed_goal_ids")
    for g in data["goals"]:
        g.pop("constraint_ids")
    data["goals"].append(goal("new:another_task"))
    schema = output_schema(SemanticSnapshot, registry=registry)
    validator = jsonschema.Draft202012Validator(schema)
    assert validator.is_valid(data)
    data["needs"][0]["goal_id"] = "new:another_task"
    assert validator.is_valid(data)
    result, candidate, _ = normalize_snapshot(
        SemanticSnapshot.model_validate(data),
        registry,
        store,
        "t2",
        profile().goal_progression.policy,
    )
    assert result.needs[0].ref == "n0001"
    assert result.needs[0].goal_id == result.goals[1].ref
    assert candidate.entries["n0001"]["scope_id"] == "g0001"
    data["needs"][0]["ref"] = "new:different_input"
    assert validator.is_valid(data)


async def test_retry_names_and_shows_the_exact_need_without_replacing_original_identity():
    c = Client({"tracker": [semantic(needs=True)]})
    s = session(client=c)
    await s.respond("Make a synthetic card.")

    def bad(call):
        value = call["context"]["previous_snapshot"]
        value["goals"].append(goal("new:another_task", event="m3"))
        value["needs"][0]["goal_id"] = "new:undeclared_task"
        return value

    def fix(call):
        payload = json.loads(call["messages"][1]["content"])
        assert "## Repair attempt" in call["messages"][0]["content"]
        excerpt = json.loads(payload["previous_output_excerpt"])
        assert excerpt["rejected_fragment"]["ref"] == "n0001"
        assert call["repair"]["errors"][0]["field_path"] == "$.needs[0].goal_id"
        return call["context"]["previous_snapshot"]

    c.script["tracker"] = [bad, fix]
    await s.respond("Keep the original input attached to its original task.")
    assert s.state.needs[0].ref == "n0001" and s.state.needs[0].goal_id == "g0001"
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "tracker:contract.reference": 1,
    }
