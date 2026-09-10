"""Context isolation, original material preservation and configured resource budgets."""

import json

import pytest
from gp_helpers import Client, goal, need, profile, ref, semantic, session

from assistant.config import GPRoleBudget
from assistant.exceptions import InvalidModelResponseError
from assistant.goal_progression.context import ContextBuilder
from assistant.goal_progression.errors import RecoveryExhausted
from assistant.goal_progression.events import EvidenceStore, IdentityIndex, InteractionIndex
from assistant.goal_progression.schemas import DeliveryUnit, SemanticSnapshot
from assistant.goal_progression.tracker import normalize_snapshot


def build_material_case():
    events = [
        {"event_id": "m1", "role": "user", "content": "ORIGINAL SOURCE: [alpha, beta, gamma]"},
        {"event_id": "m2", "role": "assistant", "content": "UNRELATED OLD CONVERSATION"},
        {"event_id": "m3", "role": "user", "content": "DEPENDENCY VALUE: curved"},
        {"event_id": "m4", "role": "assistant", "content": "ADJACENT ONLY MATERIAL"},
        {"event_id": "m5", "role": "user", "content": "Please revise the card."},
    ]
    raw = semantic()
    raw["goals"][1]["material_refs"] = [ref("m4")]
    raw["goals"][1]["evidence_refs"] = [ref("m4")]
    raw["goals"].append(goal("new:second_current", event="m3"))
    n = need("new:dep", "new:second_current", "m3")
    n.update(status="available", answer_refs=[ref("m3")])
    raw["needs"] = [n]
    store = EvidenceStore(events)
    snapshot, registry, _ = normalize_snapshot(
        SemanticSnapshot.model_validate(raw),
        IdentityIndex(),
        store,
        "t3",
        profile().goal_progression.policy,
    )
    return snapshot, ContextBuilder(store, registry, InteractionIndex())


def test_projected_contexts_omit_unrelated_history_but_preserve_original_materials():
    snapshot, builder = build_material_case()
    intra = json.dumps(builder.planner(snapshot, "intra"))
    inter = json.dumps(builder.planner(snapshot, "inter"))
    assert "ORIGINAL SOURCE: [alpha, beta, gamma]" in intra
    assert "UNRELATED OLD CONVERSATION" not in intra
    # Recency does not authorize unrelated adjacent material for Intra.
    assert "ADJACENT ONLY MATERIAL" not in intra
    assert all(g["scope"] == "current" for g in builder.planner(snapshot, "intra")["goals"])
    assert "ADJACENT ONLY MATERIAL" in inter
    assert "visible_history" not in intra + inter


def test_cross_current_dependency_closure_and_freeze_signature():
    snapshot, builder = build_material_case()
    unit = DeliveryUnit(
        goal_id=snapshot.goals[0].ref,
        target="Use the known edge shape.",
        required_need_ids=[snapshot.needs[0].ref],
        material_refs=[],
        unit_id="intra:g0001:0",
        source="intra",
        criticality="required",
    )
    from assistant.goal_progression.schemas import TurnContract

    contract = TurnContract(
        contract_id="fixture",
        snapshot_version=1,
        deliveries=[unit],
        requests=[],
        blocked_current=[],
    )
    context = builder.generator(snapshot, contract)
    assert context["needs"][0]["ref"] == snapshot.needs[0].ref
    assert "DEPENDENCY VALUE: curved" in json.dumps(context)
    assert "previous_assistant" not in context
    assert "ADJACENT ONLY MATERIAL" not in json.dumps(context)
    assert all(g["scope"] == "current" for g in context["goals"])
    original = builder.unit_signature(snapshot, unit)
    changed = snapshot.model_copy(deep=True)
    changed.goals[1].description = "A different unrelated adjacent task"
    assert builder.unit_signature(changed, unit) == original
    changed.needs[0].description = "The corrected dependency meaning"
    assert builder.unit_signature(changed, unit) != original


async def test_soft_context_expansion_is_configured_not_truncation():
    c = Client()
    s = session(client=c)
    s.settings.context.role_budgets["tracker"] = GPRoleBudget(
        max_input_tokens=1, expanded_input_tokens=32768, max_completion_tokens=8192
    )
    events = []
    await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    assert [x["role"] for x in c.calls].count("tracker") == 1
    assert s.last_call_metadata["goal_progression"]["retry_counts"]["tracker:context.capacity"] == 1
    assert c.calls[0]["context"]["visible_history"][0]["content"] == "Make a card."
    checks = [
        p for k, p in events
        if k == "assistant_gp_context_preflight" and p["role"] == "tracker"
    ]
    assert [p["status"] for p in checks] == ["role_limit", "ok"]
    assert [p["role_input_limit"] for p in checks] == [1, 32768]
    assert checks[0]["expanded"] is False and checks[1]["expanded"] is True
    assert checks[0]["input_token_estimate"] <= checks[0]["expanded_input_limit"]


async def test_required_hard_capacity_fails_without_any_model_call():
    c = Client()
    s = session(client=c)
    s.settings.context.hard_context_tokens = 1
    events = []
    with pytest.raises(RecoveryExhausted) as exc:
        await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    assert exc.value.error.code == "context.capacity"
    assert not c.calls and not s.history
    check = next(p for k, p in events if k == "assistant_gp_context_preflight")
    assert check["status"] == "hard_limit"
    assert check["hard_context_limit"] == 1
    assert check["input_token_estimate"] + check["output_token_reserve"] > 1


async def test_role_expansion_ceiling_failure_audits_limits_without_model_call():
    c = Client()
    s = session(client=c)
    s.settings.context.role_budgets["tracker"] = GPRoleBudget(
        max_input_tokens=1, expanded_input_tokens=2
    )
    events = []
    with pytest.raises(RecoveryExhausted) as exc:
        await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    check = next(p for k, p in events if k == "assistant_gp_context_preflight")
    assert check["status"] == "role_limit"
    assert check["input_token_estimate"] > check["expanded_input_limit"] == 2
    assert "configured expansion limit is 2" in exc.value.error.detail
    assert exc.value.error.recoverable is False
    assert not c.calls and not s.history


async def test_optional_context_capacity_can_prune_only_inter():
    c = Client()
    s = session(client=c)
    s.settings.context.role_budgets["inter"] = GPRoleBudget(max_input_tokens=1)
    await s.respond("Make a card.")
    assert [x["role"] for x in c.calls] == ["tracker", "intra", "generator"]
    assert s.last_call_metadata["goal_progression"]["degradations"][0]["code"] == "context.capacity"


async def test_truncated_output_growth_is_capped_by_profile():
    error = InvalidModelResponseError("truncated")
    error.call_metadata = {"finish_reason": "length", "input_tokens": 12, "output_tokens": 100}
    c = Client({"intra": [error, error, error]})
    s = session(client=c)
    s.settings.context.role_budgets["intra"].max_completion_tokens = 2500
    await s.respond("Make a card.")
    assert [x["generation"]["max_completion_tokens"] for x in c.calls if x["role"] == "intra"] == [
        2048,
        2500,
        2500,
        2500,
    ]
    assert s.last_call_metadata["goal_progression"]["retry_counts"]["intra:output.truncated"] == 3


async def test_generator_never_receives_raw_plans_history_request_ledger_or_internal_reasoning():
    c = Client(ask=True, anticipate=True)
    s = session(client=c)
    await s.respond("Make a card.")
    context = next(x["context"] for x in c.calls if x["role"] == "generator")
    encoded = json.dumps(context)
    for field in (
        "visible_history",
        "previous_snapshot",
        "intra_plan",
        "inter_plan",
        "request_eligibility",
        "issued_count",
        "SECRET_REASONING",
        "deferred",
    ):
        assert field not in encoded
    assert context["turn_contract"]["requests"][0]["question_text"] in s.history[-1].content


def test_failed_provider_usage_is_included_without_double_counting_success():
    from assistant.goal_progression.runtime import public_usage

    failed = public_usage(
        {
            "transport_attempts": [
                {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "status": "failed",
                    "raw_output": "partial",
                },
            ]
        }
    )
    assert failed["input_tokens"] == 100 and failed["output_tokens"] == 25
    successful = public_usage(
        {
            "input_tokens": 100,
            "output_tokens": 25,
            "transport_attempts": failed["transport_attempts"],
        }
    )
    assert successful["input_tokens"] == 100 and successful["output_tokens"] == 25
