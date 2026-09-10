"""Synthetic regressions for typed new identities and actionable producer repair."""

import copy
import re

import pytest
from gp_helpers import VARIANTS, Client, goal, need, profile, ref, semantic, session

from assistant.goal_progression.contracts import output_schema
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.events import EvidenceStore, IdentityIndex
from assistant.goal_progression.identity import NEW_REF_PATTERN, reference_schema
from assistant.goal_progression.prompts import load_prompts
from assistant.goal_progression.schemas import (
    Goal,
    InformationNeed,
    InterPacket,
    IntraPacket,
    SemanticSnapshot,
)
from assistant.goal_progression.tracker import normalize_snapshot


def constraint(cid="new:finish", event="m1", gid="new:current"):
    return {
        "ref": cid,
        "key": "finish",
        "value": "matte",
        "applies_to": [gid],
        "condition": None,
        "basis": "explicit",
        "evidence_refs": [ref(event)],
    }


def registered_snapshot():
    raw = semantic(adjacent=False, needs=True)
    raw["constraints"] = [constraint()]
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make a matte card."}])
    snapshot, registry, _ = normalize_snapshot(
        SemanticSnapshot.model_validate(raw),
        IdentityIndex(),
        store,
        "t1",
        profile().goal_progression.policy,
    )
    return snapshot, registry, store


def matches(schema, value):
    if "anyOf" in schema:
        return any(matches(branch, value) for branch in schema["anyOf"])
    if "enum" in schema:
        return value in schema["enum"]
    return re.fullmatch(schema["pattern"], value) is not None


@pytest.mark.parametrize("model", [SemanticSnapshot, IntraPacket, InterPacket])
def test_wire_identity_grammar_uses_actual_registry_and_kind(model):
    _, registry, _ = registered_snapshot()
    schema = output_schema(model, registry=registry)
    for name, existing, guessed, wrong_kind in (
        ("Goal", "g0001", "g0002", "c0001"),
        ("Constraint", "c0001", "c0002", "n0001"),
        ("InformationNeed", "n0001", "n0002", "g0001"),
    ):
        node = schema["$defs"][name]
        wires = [branch["properties"]["ref"] for branch in node.get("anyOf", [node])]
        assert any(matches(wire, existing) for wire in wires)
        assert any(matches(wire, "new:synthetic_name") for wire in wires)
        for wire in wires:
            assert not matches(wire, guessed) and not matches(wire, wrong_kind)
            assert not matches(wire, "bare_name") and not matches(wire, "new: ")
    for branch in schema["$defs"]["Goal"]["anyOf"]:
        assert "constraint_ids" not in branch["properties"]
    linked = schema["$defs"]["Constraint"]["properties"]["applies_to"]["items"]
    assert matches(linked, "g0001") and not matches(linked, "c0001")
    semantic_def = schema if model is SemanticSnapshot else schema["$defs"]["SemanticSnapshot"]
    assert "changed_goal_ids" not in semantic_def["properties"]


def test_packet_event_anchor_exception_is_not_a_general_identity_exception():
    registry = IdentityIndex()
    schema = output_schema(InterPacket, registry=registry, anchor_ids=["m1"])
    for branch in schema["$defs"]["Goal"]["anyOf"]:
        goal_props = branch["properties"]
        assert matches(goal_props["anchor_goal_ids"]["items"], "m1")
        assert not matches(goal_props["ref"], "m1")
        assert "constraint_ids" not in goal_props
    linked = schema["$defs"]["Constraint"]["properties"]["applies_to"]["items"]
    assert not matches(linked, "m1")


@pytest.mark.parametrize("kind", ["goal", "constraint", "need"])
def test_schema_and_allocator_share_the_same_new_ref_grammar(kind):
    _, registry, _ = registered_snapshot()
    valid = ("new:label", "new:role/local-name", "new:ABC_09.v2/part-1")
    invalid = (
        "new:",
        "new:has space",
        "g0999",
        "new:条件",
        "new:café",
        "new:🙂",
        "new:a\u00a0b",
        "new:a\u2003b",
        "new:a\u200bb",
        "new:a\u3000b",
        "new:a\nb",
        "new:a\tb",
        "new:label\n",
        'new:a"b',
        "new:a\\b",
        "new:a:b",
    )
    for value in (*valid, *invalid):
        allowed = matches(reference_schema(registry, kind), value)
        assert allowed is (value in valid)
        candidate = registry.clone()
        if allowed:
            assert candidate.allocate(value, kind, "m1", "t2", "tracker") in candidate.entries
        else:
            with pytest.raises(ContractError):
                candidate.allocate(value, kind, "m1", "t2", "tracker")


@pytest.mark.parametrize("model", [SemanticSnapshot, IntraPacket, InterPacket])
def test_every_identity_pattern_uses_a_positive_ascii_class(model):
    _, registry, _ = registered_snapshot()
    patterns = []

    def visit(node):
        if isinstance(node, dict):
            if "pattern" in node:
                patterns.append(node["pattern"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(output_schema(model, registry=registry))
    assert patterns and set(patterns) == {r"^new:[A-Za-z0-9_./-]+$"}
    assert NEW_REF_PATTERN == patterns[0]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_identity_producer_prompt_describes_the_ascii_name_contract(variant):
    prompts = load_prompts(profile(variant))
    producers = ("intra", "inter") if variant == "no_tracker" else ("tracker",)
    mode = "no_tracker" if variant == "no_tracker" else "normal"
    for role in producers:
        text = prompts[role].system_for(mode)
        assert "ASCII letters" in text and "digits, _, ., /, -" in text


async def test_ascii_ids_do_not_restrict_unicode_semantics_materials_or_reply():
    raw = semantic(adjacent=False)
    raw["goals"][0].update(description="制作一张中文卡片", remaining_work="输出完整卡片内容")
    raw["constraints"] = [constraint()]
    raw["constraints"][0].update(key="表面", value="磨砂", condition="本轮使用蓝色")
    body = "蓝色卡片：你好，世界！🙂"

    def realize(call):
        assert call["context"]["goals"][0]["description"] == "制作一张中文卡片"
        assert call["context"]["constraints"][0]["value"] == "磨砂"
        assert call["context"]["latest_user"]["content"] == "请制作中文磨砂卡片。"
        return {
            "units": [
                {"unit_id": item["unit_id"], "text": body}
                for item in call["context"]["turn_contract"]["deliveries"]
            ],
            "issues": [],
        }

    s = session(client=Client({"tracker": [raw], "generator": [realize]}))
    assert await s.respond("请制作中文磨砂卡片。") == body
    assert s.state.goals[0].ref == "g0001"
    assert s.state.constraints[0].condition == "本轮使用蓝色"


async def test_non_ascii_new_id_uses_existing_owner_retry_without_silent_renaming():
    bad = semantic(adjacent=False, current="new:条件")

    def repaired(call):
        error = call["repair"]["errors"][0]
        assert error["field_path"] == "$.current_goals[0].ref"
        assert "ASCII letters, digits, _, ., /, -" in error["detail"]
        return semantic(adjacent=False)

    s = session(client=Client({"tracker": [bad, repaired]}))
    await s.respond("制作一张卡片。")
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "tracker:contract.reference": 1
    }


def test_same_output_allocation_cannot_legitimize_a_guessed_canonical_id():
    raw = semantic(adjacent=False)
    raw["goals"].append(goal("g0001"))
    registry = IdentityIndex()
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make two cards."}])
    with pytest.raises(ContractError) as error:
        normalize_snapshot(
            SemanticSnapshot.model_validate(raw),
            registry,
            store,
            "t1",
            profile().goal_progression.policy,
        )
    assert error.value.field_path == "$.goals[1].ref"
    assert not registry.entries and not registry.aliases


def test_cross_kind_reference_is_rejected_even_if_id_exists():
    snapshot, registry, store = registered_snapshot()
    snapshot.needs[0].goal_id = "c0001"
    with pytest.raises(ContractError) as error:
        normalize_snapshot(snapshot, registry, store, "t2", profile().goal_progression.policy)
    assert error.value.field_path == "$.needs[0].goal_id"
    assert "Registered goal IDs: ['g0001']" in error.value.detail


async def test_second_turn_repair_explains_new_id_and_preserves_registered_state():
    initial = semantic(adjacent=False)
    initial["constraints"] = [constraint()]
    c = Client({"tracker": [initial]})
    s = session(client=c)
    await s.respond("Make a matte card.")

    def wrong(call):
        value = copy.deepcopy(call["context"]["previous_snapshot"])
        value["constraints"].append(constraint("c0002", "m3", "g0001"))
        return value

    def corrected(call):
        context = call["context"]
        assert context["registered_ids"] == {"goal": ["g0001"], "constraint": ["c0001"], "need": []}
        error = call["repair"]["errors"][0]
        assert error["field_path"] == "$.constraints[1].ref"
        assert "'c0002'" in error["detail"] and "new:<local_name>" in error["detail"]
        assert "Registered constraint IDs: ['c0001']" in error["detail"]
        wire = call["schema"]["$defs"]["Constraint"]["properties"]["ref"]
        assert not matches(wire, "c0002")
        value = wrong(call)
        value["constraints"][-1]["ref"] = "new:accent"
        return value

    c.script["tracker"] = [wrong, corrected]
    await s.respond("Keep the finish and add a blue accent.")
    assert s.state.goals[0].ref == "g0001"
    assert [item.ref for item in s.state.constraints] == ["c0001", "c0002"]
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "tracker:contract.reference": 1
    }


def test_need_applicability_does_not_merge_new_identity_or_rewrite_origin():
    snapshot, registry, store = registered_snapshot()
    snapshot.goals.append(Goal.model_validate(goal("new:second_task")))
    snapshot.needs.append(
        InformationNeed.model_validate(need("new:second_shape", "new:second_task"))
    )
    snapshot.needs[0].goal_id = "new:second_task"
    result, candidate, _ = normalize_snapshot(
        snapshot, registry, store, "t2", profile().goal_progression.policy
    )
    assert result.needs[0].ref == "n0001"
    assert result.needs[0].goal_id == result.goals[1].ref
    assert result.needs[1].ref != result.needs[0].ref
    assert candidate.entries["n0001"]["scope_id"] == "g0001"
    assert candidate.entries[result.needs[1].ref]["scope_id"] == result.goals[1].ref
