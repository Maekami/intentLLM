"""Continuity follows selected original evidence, not automatic reply broadcast."""

import json

from test_gp_context_v2 import build_material_case

from assistant.goal_progression.schemas import DeliveryUnit, EvidenceRef, TurnContract


def fixture_unit(snapshot):
    return DeliveryUnit(
        goal_id=snapshot.goals[0].ref,
        target="Revise the card.",
        required_need_ids=[],
        material_refs=[],
        unit_id="intra:g0001:0",
        source="intra",
        criticality="required",
    )


def test_previous_reply_is_scoped_by_material_not_recency():
    snapshot, builder = build_material_case()
    intra = builder.planner(snapshot, "intra")
    inter = builder.planner(snapshot, "inter")
    assert "previous_assistant" not in intra and "previous_assistant" not in inter
    assert "ADJACENT ONLY MATERIAL" not in json.dumps(intra)
    assert "ADJACENT ONLY MATERIAL" in json.dumps(inter)
    assert "UNRELATED OLD CONVERSATION" not in json.dumps([intra, inter])


def test_selected_previous_reply_block_is_visible_and_not_dropped_by_deduplication():
    snapshot, builder = build_material_case()
    builder.store.events[-2]["blocks"] = {"draft": {"start": 0, "end": 8}}
    snapshot.goals[0].material_refs = [EvidenceRef(event_id="m4", block_id="draft")]
    context = builder.planner(snapshot, "intra")
    domain = builder.evidence_domain(context)
    assert domain["m4"] == ["draft"]
    assert "m2" not in domain
    projected = next(e for e in context["evidence"] if e["event_id"] == "m4")
    assert projected["content"] == builder.store.events[-2]["content"][:8]
    projected["content"] = "Changed projected copy"
    assert builder.store.events[-2]["content"] == "ADJACENT ONLY MATERIAL"


def test_latest_user_is_preserved_without_duplicate_whole_message():
    snapshot, builder = build_material_case()
    snapshot.goals[0].material_refs.append(EvidenceRef(event_id="m5"))
    context = builder.planner(snapshot, "intra")
    assert context["latest_user"]["content"] == "Please revise the card."
    assert all(e["event_id"] != "m5" for e in context["evidence"])
    assert builder.evidence_domain(context)["m5"] == [None]


def test_generator_keeps_explicit_unit_material_from_previous_reply():
    snapshot, builder = build_material_case()
    unit = fixture_unit(snapshot)
    unit.material_refs = [EvidenceRef(event_id="m4")]
    contract = TurnContract(
        contract_id="fixture",
        snapshot_version=1,
        deliveries=[unit],
        requests=[],
        blocked_current=[],
    )
    context = builder.generator(snapshot, contract)
    assert "previous_assistant" not in context
    assert "ADJACENT ONLY MATERIAL" in json.dumps(context["evidence"])
    assert builder.evidence_domain(context)["m4"] == [None]


def test_unrelated_previous_reply_does_not_invalidate_frozen_current_unit():
    snapshot, builder = build_material_case()
    unit = fixture_unit(snapshot)
    original = builder.unit_signature(snapshot, unit)
    builder.store.events[-2]["content"] = "An unrelated adjacent update."
    assert builder.unit_signature(snapshot, unit) == original


def test_changed_explicitly_selected_previous_material_invalidates_frozen_unit():
    snapshot, builder = build_material_case()
    unit = fixture_unit(snapshot)
    snapshot.goals[0].material_refs = [EvidenceRef(event_id="m4")]
    original = builder.unit_signature(snapshot, unit)
    builder.store.events[-2]["content"] = "Corrected original material."
    assert builder.unit_signature(snapshot, unit) != original
