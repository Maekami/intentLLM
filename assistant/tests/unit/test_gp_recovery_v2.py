"""Independent quotas, producer repair and bounded degradation (synthetic only)."""

import asyncio

import pytest
from gp_helpers import Client, delivery, profile, semantic, session
from pydantic import ValidationError

from assistant.config import GP_ERROR_CODES, GPRecoverySettings
from assistant.exceptions import ConfigurationError
from assistant.goal_progression.errors import ContractError, RecoveryExhausted, RetryLedger


def test_every_error_has_its_own_three_retries_and_owner():
    ledger = RetryLedger(profile().goal_progression)
    for owner in ("tracker", "intra", "inter", "joint", "generator", "runtime"):
        for code in GP_ERROR_CODES:
            for i in range(3):
                ledger.consume([ContractError(code, owner, "variable detail " + str(i))])
            with pytest.raises(RecoveryExhausted):
                ledger.consume([ContractError(code, owner, "different field/version")])
    assert set(ledger.counts.values()) == {3}


def test_multi_error_retry_is_atomic_and_overrides_are_bounded():
    settings = profile().goal_progression
    settings.recovery = GPRecoverySettings(by_role_and_code={"intra": {"output.parse": 0}})
    ledger = RetryLedger(settings)
    with pytest.raises(RecoveryExhausted):
        ledger.consume(
            [
                ContractError("output.schema", "inter", "x"),
                ContractError("output.parse", "intra", "x"),
            ]
        )
    assert ledger.counts == {}
    for raw in (
        {"default_max_retries": 4},
        {"by_role_and_code": {"intra": {"output.parse": 4}}},
        {"by_role_and_code": {"other": {"output.parse": 1}}},
        {"by_role_and_code": {"intra": {"free_text_error": 1}}},
    ):
        with pytest.raises(ValidationError):
            GPRecoverySettings.model_validate(raw)


async def test_timeout_parse_schema_are_independent_and_parallel_result_is_preserved():
    c = Client({"intra": [TimeoutError(), "not-json", {}, "not-json"]})
    s = session(client=c)
    await s.respond("Make a card.")
    assert [x["role"] for x in c.calls].count("tracker") == 1
    assert [x["role"] for x in c.calls].count("inter") == 1
    assert [x["role"] for x in c.calls].count("intra") == 5
    assert c.max_parallel_planners == 2
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "intra:call.timeout": 1,
        "intra:output.parse": 2,
        "intra:output.schema": 1,
    }


async def test_timeout_audit_does_not_claim_a_received_output_or_semantic_failure():
    c = Client({"generator": [TimeoutError()]})
    s = session(client=c)
    events = []
    await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    failed = next(p for k, p in events if k == "assistant_generator_failed")
    assert failed["response_received"] is False
    assert failed["call_timeout_seconds"] == s.settings.recovery.call_timeout_seconds
    assert "raw_output" not in failed
    gen_calls = [call for call in c.calls if call["role"] == "generator"]
    assert "not evidence" in gen_calls[1]["repair"]["instruction"]
    received = [p for k, p in events if k == "assistant_generator_raw_result"]
    assert received[-1]["response_received"] is True


async def test_optional_planner_exhausts_exactly_four_attempts():
    c = Client({"inter": ["not-json"] * 4})
    s = session(client=c)
    await s.respond("Make a card.")
    assert [x["role"] for x in c.calls].count("inter") == 4
    assert [x["role"] for x in c.calls].count("intra") == 1
    assert s.last_call_metadata["goal_progression"]["degradations"][0]["owner"] == "inter"


@pytest.mark.parametrize("role", ["tracker", "intra", "generator"])
async def test_required_producer_never_falls_back_to_prompted_base(role):
    c = Client({role: ["not-json"] * 4})
    s = session(client=c)
    events = []
    with pytest.raises(RecoveryExhausted):
        await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    assert not s.history and s.state is None and not s.interaction_state
    assert [x["role"] for x in c.calls].count(role) == 4
    assert events[-1][0] == "assistant_gp_turn_failed"
    assert s.last_call_metadata["visible_response_tokens"] == 0
    assert not any("fallback" in k for k, _ in events)


def issue(call, *, source="intra", owner="unit"):
    units, issues = [], []
    for u in call["context"]["turn_contract"]["deliveries"]:
        if u["source"] == source:
            reference = "unit:" + u["unit_id"] if owner == "unit" else "goal:" + u["goal_id"]
            issues.append(
                {
                    "code": "realization.unrealizable",
                    "unit_id": u["unit_id"],
                    "input_refs": [reference],
                    "detail": "The specified target needs revision.",
                }
            )
        else:
            units.append({"unit_id": u["unit_id"], "text": "FROZEN independent artifact."})
    return {"units": units, "issues": issues}


async def test_late_current_issue_repairs_only_its_planner_and_retains_adjacent_body():
    c = Client({"generator": [issue]}, anticipate=True)
    s = session(client=c)
    events = []
    reply = await s.respond("Make a card.", audit_sink=lambda k, p: events.append((k, p)))
    roles = [x["role"] for x in c.calls]
    assert roles.count("intra") == 2 and roles.count("inter") == 1 and roles.count("tracker") == 1
    assert roles.count("generator") == 2 and "FROZEN independent artifact." in reply
    last = c.calls[-1]["context"]
    assert len(last["turn_contract"]["deliveries"]) == 1
    assert last["frozen_body_for_continuity"][0]["text"] == "FROZEN independent artifact."
    assert any(k == "assistant_gp_dependencies_invalidated" for k, _ in events)
    repair = next(x["repair"] for x in c.calls if x["role"] == "intra" and x["repair"])
    assert "previous_owned_output_excerpt" in repair


async def test_tracker_issue_reprojects_dependencies_without_resetting_retry_budget():
    c = Client({"generator": [lambda x: issue(x, owner="goal")]})
    s = session(client=c)
    await s.respond("Make a card.")
    roles = [x["role"] for x in c.calls]
    assert roles.count("tracker") == roles.count("intra") == roles.count("inter") == 2
    repair_tracker = [x for x in c.calls if x["role"] == "tracker"][1]
    assert repair_tracker["context"]["previous_snapshot"]["goals"][0]["ref"] == "g0001"
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {
        "tracker:realization.unrealizable": 1,
    }


async def test_optional_issue_repair_failure_preserves_required_body_without_rewriting():
    c = Client(anticipate=True)
    c.script = {
        "inter": [lambda x: c.default(x), *["invalid"] * 4],
        "generator": [lambda x: issue(x, source="inter")],
    }
    s = session(client=c)
    assert await s.respond("Make a card.") == "FROZEN independent artifact."
    assert [x["role"] for x in c.calls].count("generator") == 1
    assert s.last_call_metadata["goal_progression"]["degradations"]


async def test_joint_optional_repair_failure_retains_previously_valid_current_plan():
    c = Client()

    def joint(call):
        value = c.default(call)
        gid = call["context"]["adjacent_goal_ids"][0]
        value["inter"] = {
            "action": "anticipate",
            "goal_id": gid,
            "deliveries": [delivery(gid)],
            "request": None,
        }
        return value

    c.script = {
        "joint": [joint, *["invalid"] * 4],
        "generator": [lambda x: issue(x, source="inter")],
    }
    s = session("joint", client=c)
    assert await s.respond("Make a card.") == "FROZEN independent artifact."
    assert [x["role"] for x in c.calls].count("joint") == 5
    assert [x["role"] for x in c.calls].count("generator") == 1


async def test_no_tracker_bad_optional_packet_does_not_spin_after_pruning():
    value = semantic()
    value["goals"] = [value["goals"][1]]
    value["goals"][0]["anchor_goal_ids"] = ["missing-anchor"]
    c = Client(
        {
            "inter": [
                {
                    "semantic": value,
                    "plan": {
                        "action": "none",
                        "goal_id": None,
                        "deliveries": [],
                        "request": None,
                    },
                }
            ]
            * 4
        }
    )
    s = session("no_tracker", client=c)
    await asyncio.wait_for(s.respond("Make a card."), timeout=3)
    assert [x["role"] for x in c.calls].count("intra") == 1
    assert s._version == 1


async def test_fatal_configuration_is_not_retried():
    c = Client({"tracker": [ConfigurationError("bad credentials")]})
    s = session(client=c)
    with pytest.raises(ConfigurationError):
        await s.respond("Make a card.")
    assert len(c.calls) == 1


async def test_nested_provider_retries_are_rejected():
    p = profile()
    p.retry.max_attempts = 2
    from assistant.goal_progression import GoalProgressionBaseline, GoalProgressionSession

    with pytest.raises(ConfigurationError):
        GoalProgressionSession(
            Client(), p.generation["assistant"], GoalProgressionBaseline(), profile=p
        )


async def test_no_intra_optional_repair_reuses_the_accepted_current_plan_and_body():
    c = Client(anticipate=True)

    def realize_with_issue(call):
        result = c.default(call)
        adjacent = call["context"]["inter_work"]["deliveries"][0]
        result["units"] = [u for u in result["units"] if not u["unit_id"].startswith("inter:")]
        result["units"][0]["text"] = "FROZEN no_intra card."
        result["issues"] = [
            {
                "code": "realization.unrealizable",
                "unit_id": adjacent["unit_id"],
                "input_refs": ["unit:" + adjacent["unit_id"]],
                "detail": "Revise target.",
            }
        ]
        return result

    c.script = {
        "inter": [lambda x: c.default(x), *["invalid"] * 4],
        "generator": [realize_with_issue],
    }
    s = session("no_intra", client=c)
    assert await s.respond("Make a card.") == "FROZEN no_intra card."
    assert [x["role"] for x in c.calls].count("generator") == 1


async def test_cancelled_inflight_turn_does_not_commit_and_cleans_up_planners():
    began = asyncio.Event()

    class WaitingClient(Client):
        async def generate(self, **kwargs):
            began.set()
            await asyncio.Event().wait()

    s = session(client=WaitingClient())
    task = asyncio.create_task(s.respond("Make a card."))
    await began.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not s.history and not s._lock.locked()
    assert s.last_call_metadata["goal_progression"]["termination_reason"] == "cancelled"


async def test_deadline_is_not_misreported_as_shared_retry_exhaustion():
    class WaitingClient(Client):
        async def generate(self, **kwargs):
            await asyncio.Event().wait()

    s = session(client=WaitingClient(), turn_timeout_seconds=0.05)
    with pytest.raises(TimeoutError):
        await s.respond("Make a card.")
    assert s.last_call_metadata["goal_progression"]["termination_reason"] == "turn_deadline"
    assert s.last_call_metadata["goal_progression"]["retry_counts"] == {}


async def test_optional_failure_with_only_a_required_request_needs_no_extra_generator():
    c = Client(ask=True, anticipate=True)

    def question_only(call):
        value = c.default(call)
        value["items"][0]["deliveries"] = []
        return value

    c.script = {
        "intra": [question_only],
        "inter": [lambda x: c.default(x), *["invalid"] * 4],
        "generator": [lambda x: issue(x, source="inter")],
    }
    s = session(client=c)
    assert await s.respond("Make a card.") == "Which edge shape should the card use?"
    assert [x["role"] for x in c.calls].count("generator") == 1
