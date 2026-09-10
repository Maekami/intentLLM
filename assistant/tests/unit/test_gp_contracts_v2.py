"""Role boundaries, configurable request policy and thin body realization."""

import pytest
from gp_helpers import Client, current_plan, delivery, need, profile, semantic, session

from assistant.goal_progression.contracts import output_schema, validate_inter, validate_intra
from assistant.goal_progression.errors import ContractError, RecoveryExhausted
from assistant.goal_progression.events import EvidenceStore, IdentityIndex, InteractionIndex
from assistant.goal_progression.schemas import (
    CurrentRealization,
    GeneratorOutput,
    InterPlan,
    IntraPlan,
    JointPlan,
    SemanticSnapshot,
)
from assistant.goal_progression.tracker import normalize_snapshot


def normalized(*, needs=True):
    raw = semantic(needs=needs)
    n = need("new:label", "new:adjacent")
    raw["needs"].append(n)
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make a card."}])
    snapshot, registry, _ = normalize_snapshot(
        SemanticSnapshot.model_validate(raw),
        IdentityIndex(),
        store,
        "t1",
        profile().goal_progression.policy,
    )
    eligibility = InteractionIndex().view(snapshot, registry, store)
    return snapshot, store, eligibility


def test_wire_schema_limits_ids_coverage_and_variant_action():
    snapshot, _, _ = normalized()
    wire = output_schema(IntraPlan, snapshot=snapshot)
    assert wire["properties"]["items"]["minItems"] == wire["properties"]["items"]["maxItems"] == 1
    assert wire["additionalProperties"] is False
    for branch in wire["$defs"]["IntraItem"]["anyOf"]:
        assert branch["properties"]["goal_id"]["enum"] == ["g0001"]
        assert set(branch["required"]) == set(branch["properties"])
    inter = output_schema(InterPlan, snapshot=snapshot, variant="no_anticipate")
    assert [b["properties"]["action"]["const"] for b in inter["anyOf"]] == ["none", "elicit"]
    empty = output_schema(GeneratorOutput, unit_ids=[])
    assert empty["properties"]["units"]["maxItems"] == 0


@pytest.mark.parametrize(
    "model,ids",
    [
        (IntraPlan, ["n0001"]),
        (InterPlan, ["n0002"]),
        (JointPlan, ["n0001", "n0002"]),
        (CurrentRealization, ["n0001"]),
    ],
)
def test_planner_reference_domains_are_scoped_and_never_free_text(model, ids):
    snapshot, _, eligibility = normalized()
    wire = output_schema(model, snapshot=snapshot, request_eligibility=eligibility)
    defs = wire["$defs"]
    assert defs["RequestProposal"]["properties"]["need_id"]["enum"] == ids
    assert defs["DeliveryProposal"]["properties"]["required_need_ids"]["maxItems"] == 0
    if "IntraItem" in defs:
        for branch in defs["IntraItem"]["anyOf"]:
            assert branch["properties"]["blocked_by"]["items"]["enum"] == ids


def test_known_request_facts_remove_ineligible_request_branch_from_wire():
    snapshot, _, eligibility = normalized()
    eligibility["n0001"]["request_eligible"] = False
    wire = output_schema(IntraPlan, snapshot=snapshot, request_eligibility=eligibility)
    for branch in wire["$defs"]["IntraItem"]["anyOf"]:
        assert branch["properties"]["request"] == {"type": "null"}
        assert branch["properties"]["action"]["const"] != "clarify"
    snapshot.needs[0].status = "available"
    wire = output_schema(IntraPlan, snapshot=snapshot)
    for branch in wire["$defs"]["IntraItem"]["anyOf"]:
        assert branch["properties"]["blocked_by"]["maxItems"] == 0
        assert branch["properties"]["action"]["const"] in {"advance", "revise"}
    assert wire["$defs"]["DeliveryProposal"]["properties"]["required_need_ids"]["items"][
        "enum"
    ] == ["n0001"]


async def test_blocked_reference_with_prose_is_repaired_by_its_owner():
    c = Client(ask=True)

    def blocked(call):
        result = c.default(call)
        item = result["items"][0]
        item.update(action=None, deliveries=[], request=None, blocked_by=["n0001: missing shape"])
        return result

    def corrected(call):
        error = call["repair"]["errors"][0]
        assert error["field_path"] == "$.items[0].blocked_by[0]"
        assert "exact Need IDs only: ['n0001']" in error["detail"]
        value = blocked(call)
        value["items"][0]["blocked_by"] = ["n0001"]
        return value

    c.script["intra"] = [blocked, corrected]
    s = session(client=c)
    await s.respond("Make a card.")
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "intra:contract.dependency": 1
    }


@pytest.mark.parametrize(
    "kind",
    ["missing_goal", "wrong_scope", "bad_dependency", "unavailable_dependency", "bad_request"],
)
def test_machine_contract_rejects_invalid_current_policy(kind):
    snapshot, store, eligibility = normalized()
    plan = IntraPlan.model_validate(current_plan(snapshot.model_dump()))
    if kind == "missing_goal":
        plan.items = []
    elif kind == "wrong_scope":
        plan.items[0].goal_id = "g0002"
    elif kind == "bad_dependency":
        plan.items[0].deliveries[0].required_need_ids = ["not_registered"]
    elif kind == "unavailable_dependency":
        plan.items[0].deliveries[0].required_need_ids = ["n0001"]
    else:
        plan.items[0].action = "clarify"
        from assistant.goal_progression.schemas import RequestProposal

        plan.items[0].request = RequestProposal(need_id="n0002", question_text="Which label?")
    with pytest.raises(ContractError):
        validate_intra(plan, snapshot, eligibility, store)


def test_inter_cannot_anticipate_a_current_goal_or_ask_unregistered_need():
    snapshot, store, eligibility = normalized()
    for raw in (
        {
            "action": "anticipate",
            "goal_id": "g0001",
            "deliveries": [delivery("g0001")],
            "request": None,
        },
        {
            "action": "elicit",
            "goal_id": "g0002",
            "deliveries": [],
            "request": {"need_id": "n0001", "question_text": "Which shape?"},
        },
    ):
        with pytest.raises(ContractError):
            validate_inter(
                InterPlan.model_validate(raw),
                snapshot,
                eligibility,
                store,
                profile().goal_progression.policy,
            )


@pytest.mark.parametrize("variant", ["full", "no_intra"])
@pytest.mark.parametrize("budget", [0, 1, 2])
async def test_current_first_frozen_requests_obey_configured_budget(variant, budget):
    raw = semantic(needs=True)
    raw["needs"].append(need("new:label", "new:adjacent"))

    def elicit(call):
        n = next(n for n in call["context"]["needs"] if n["goal_id"] == "g0002")
        return {
            "action": "elicit",
            "goal_id": "g0002",
            "deliveries": [],
            "request": {"need_id": n["ref"], "question_text": "Which storage label?"},
        }

    c = Client({"tracker": [raw], "inter": [elicit]}, ask=True)
    s = session(variant, client=c, policy={"request_budget": budget})
    reply = await s.respond("Make a card.")
    requests = s.last_call_metadata["goal_progression"]["receipt"]["issued_requests"]
    assert [r["need_id"] for r in requests] == ["n0001", "n0002"][:budget]
    assert ("Which edge shape" in reply) is (budget >= 1)
    assert ("Which storage label" in reply) is (budget >= 2)


async def test_artifact_question_marks_are_not_treated_as_new_clarification():
    def questionnaire(call):
        return {
            "units": [
                {"unit_id": u["unit_id"], "text": "Survey draft: 1. Which shape? 2. Which color?"}
                for u in call["context"]["turn_contract"]["deliveries"]
            ],
            "issues": [],
        }

    s = session(client=Client({"generator": [questionnaire]}))
    assert "Which color?" in await s.respond("Draft a synthetic survey.")
    assert not s.interaction_state
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {}


async def test_blank_unit_repair_preserves_already_valid_unit():
    def partial(call):
        left, right = call["context"]["turn_contract"]["deliveries"]
        return {
            "units": [
                {"unit_id": left["unit_id"], "text": "PRESERVED current card."},
                {"unit_id": right["unit_id"], "text": "   "},
            ],
            "issues": [],
        }

    c = Client({"generator": [partial]}, anticipate=True)
    s = session(client=c)
    reply = await s.respond("Make a card.")
    assert reply.startswith("PRESERVED current card.")
    assert len(c.calls[-1]["context"]["turn_contract"]["deliveries"]) == 1
    assert (
        s.last_call_metadata["goal_progression"]["retry_counts"]["generator:contract.coverage"] == 1
    )


async def test_unknown_generator_unit_cannot_be_delivered():
    bad = {"units": [{"unit_id": "unapproved", "text": "Not authorized."}], "issues": []}
    s = session(client=Client({"generator": [bad] * 4}))
    with pytest.raises(RecoveryExhausted):
        await s.respond("Make a card.")
    assert not s.history


async def test_no_intra_question_only_discards_unneeded_boundary_issue():
    c = Client(ask=True)

    def question_only(call):
        result = c.default(call)
        result["intra"]["items"][0]["deliveries"] = []
        result["units"] = []
        gid = result["intra"]["items"][0]["goal_id"]
        result["issues"] = [
            {
                "code": "context.missing",
                "unit_id": "boundary:" + gid,
                "input_refs": ["goal:" + gid],
                "detail": "No boundary is needed.",
            }
        ]
        return result

    c.script = {"generator": [question_only]}
    s = session("no_intra", client=c)
    assert await s.respond("Make a card.") == "Which edge shape should the card use?"
    assert len(c.calls) == 3


@pytest.mark.parametrize("variant", ["full", "no_tracker", "no_inter", "joint", "no_anticipate"])
async def test_request_only_contract_never_calls_generator(variant):
    c = Client(ask=True)

    def question_only(call):
        result = c.default(call)
        plan = (
            result["plan"]
            if variant == "no_tracker"
            else result["intra"]
            if variant == "joint"
            else result
        )
        plan["items"][0]["deliveries"] = []
        return result

    role = "joint" if variant == "joint" else "intra"
    c.script[role] = [question_only]
    c.script["generator"] = [AssertionError("An empty body work order must not call a model")]
    s = session(variant, client=c)
    assert await s.respond("Make a card.") == "Which edge shape should the card use?"
    audit = s.last_call_metadata["goal_progression"]
    assert not any(call["role"] == "generator" for call in c.calls)
    assert audit["retry_counts"] == {}
    assert audit["receipt"]["delivered_units"] == []
    assert len(audit["receipt"]["issued_requests"]) == 1


async def test_unselected_request_still_requires_a_boundary_body():
    c = Client(ask=True)

    def question_only(call):
        result = c.default(call)
        result["items"][0]["deliveries"] = []
        return result

    c.script["intra"] = [question_only]
    s = session(client=c, policy={"request_budget": 0})
    await s.respond("Make a card.")
    assert c.calls[-1]["role"] == "generator"
    assert c.calls[-1]["context"]["turn_contract"]["deliveries"][0]["kind"] == "boundary"
    assert not s.last_call_metadata["goal_progression"]["receipt"]["issued_requests"]


async def test_omitting_a_required_wire_field_does_not_silently_fill_a_default():
    bad = semantic()
    del bad["goals"][0]["material_refs"]
    s = session(client=Client({"tracker": [bad]}))
    await s.respond("Make a card.")
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {"tracker:output.schema": 1}


def test_generator_schema_bounds_entries_and_exposes_only_known_issue_refs():
    wire = output_schema(
        GeneratorOutput,
        unit_ids=["intra:g0001:0"],
        input_reference_ids=["unit:intra:g0001:0", "goal:g0001"],
    )
    assert wire["properties"]["units"]["maxItems"] == 1
    assert wire["properties"]["issues"]["maxItems"] == 1
    issue = wire["$defs"]["RealizationIssue"]["properties"]
    assert issue["input_refs"]["items"]["enum"] == ["unit:intra:g0001:0", "goal:g0001"]


async def test_body_issue_overlap_repair_names_the_conflict_and_required_choice():
    c = Client()

    def overlap(call):
        result = c.default(call)
        uid = result["units"][0]["unit_id"]
        result["issues"] = [
            {
                "code": "realization.unrealizable",
                "unit_id": uid,
                "input_refs": ["unit:" + uid],
                "detail": "A synthetic conflict.",
            }
        ]
        return result

    def corrected(call):
        error = call["repair"]["errors"][0]
        assert error["affected_units"] == ["intra:g0001:0"]
        assert "BOTH" in error["detail"] and "COMPLETE body" in error["detail"]
        assert "frozen_body_for_continuity" not in call["context"]
        return c.default(call)

    c.script["generator"] = [overlap, corrected]
    s = session(client=c)
    await s.respond("Make a card.")
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "generator:contract.reference": 1
    }


@pytest.mark.parametrize("mode", ["schema", "prompt"])
async def test_decoding_mode_changes_only_provider_constraint_not_local_validation(mode):
    c = Client(
        {"generator": [{"units": [{"unit_id": "unapproved", "text": "Invalid."}], "issues": []}]}
    )
    s = session(client=c, structured_decoding={"generator": mode})
    events = []
    await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    gen_calls = [call for call in c.calls if call["role"] == "generator"]
    assert len(gen_calls) == 2
    assert (gen_calls[0]["schema"] is not None) is (mode == "schema")
    assert "## Output schema" in gen_calls[0]["messages"][0]["content"]
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "generator:contract.reference": 1
    }
    requested = next(p for k, p in events if k == "assistant_generator_requested")
    assert requested["structured_decoding"] == mode
    assert requested["response_schema_sent"] is (mode == "schema")
    assert requested["response_schema"]["properties"]["units"]["maxItems"] == 1
    assert all(call["schema"] is not None for call in c.calls if call["role"] != "generator")


@pytest.mark.parametrize("setting", [{"runtime": "prompt"}, {"generator": "unchecked"}])
def test_decoding_modes_only_accept_known_model_roles_and_supported_modes(setting):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        profile(structured_decoding=setting)
