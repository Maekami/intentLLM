"""Transactional state, exact renderer receipts and historical identity boundaries."""

import copy

import pytest
from gp_helpers import Client, goal, need, profile, ref, semantic, session

from assistant.goal_progression.errors import RecoveryExhausted
from assistant.goal_progression.events import (
    EvidenceStore,
    IdentityIndex,
    text_digest,
)
from assistant.goal_progression.schemas import SemanticSnapshot
from assistant.goal_progression.tracker import normalize_snapshot


async def test_receipt_spans_hashes_and_idempotent_actual_issuance():
    s = session(client=Client(ask=True))
    reply = await s.respond("Make a card.")
    receipt = s.last_call_metadata["goal_progression"]["receipt"]
    assert receipt["reply_hash"] == text_digest(reply)
    for part in [*receipt["delivered_units"], *receipt["issued_requests"]]:
        span = part["text_span"]
        assert part["text_hash"] == text_digest(reply[span["start"] : span["end"]])
    assert len(receipt["issued_requests"]) == 1
    before = s.interaction_state
    s._interactions.apply(receipt, 1)
    assert s.interaction_state == before
    assert before["n0001"]["issued_count"] == 1
    assert s.state.goals[0].status == "active"  # structural delivery is not semantic completion
    assert s.last_call_metadata["visible_response_tokens"] == len(reply.split())
    assert s.last_call_metadata["internal_output_tokens"] > len(reply.split())


async def test_renewed_authorization_is_consumed_once_by_actual_return():
    c = Client(ask=True)
    s = session(client=c)
    await s.respond("Make a card.")

    def renewed(call):
        value = copy.deepcopy(call["context"]["previous_snapshot"])
        value["needs"][0]["renewed_authorization_ref"] = ref("m3")
        return value

    c.script["tracker"] = [renewed]
    await s.respond("You can ask me again.")
    assert s.interaction_state["n0001"]["issued_count"] == 2
    await s.respond("I still cannot answer.")
    assert s.interaction_state["n0001"]["issued_count"] == 2
    assert c.calls[-1]["context"]["turn_contract"]["requests"] == []


@pytest.mark.parametrize(
    "failure_at",
    [
        "assistant_gp_commit_prepared",
        "assistant_gp_state_committed",
    ],
)
async def test_audit_failure_before_swap_rolls_back_all_visible_and_request_state(failure_at):
    s = session(client=Client(ask=True))
    recorded = []

    def sink(kind, payload):
        recorded.append((kind, payload))
        if kind == failure_at:
            raise OSError("disk unavailable")

    with pytest.raises(RecoveryExhausted):
        await s.respond("Make a card.", audit_sink=sink)
    assert not s.history and s.state is None and s.interaction_state == {}
    assert s.last_call_metadata["goal_progression"]["committed"] is False
    failed_writes = [p["event_id"] for k, p in recorded if k == failure_at]
    assert len(failed_writes) == 4 and len(set(failed_writes)) == 1


async def test_temporary_audit_retry_does_not_rerun_a_model_or_double_issue():
    c, attempts = Client(ask=True), []
    s = session(client=c)

    def sink(kind, payload):
        if kind == "assistant_gp_state_committed":
            attempts.append(payload["event_id"])
            if len(attempts) < 3:
                raise OSError("temporary failure")

    await s.respond("Make a card.", audit_sink=sink)
    assert len(c.calls) == 4 and len(set(attempts)) == 1
    assert s.interaction_state["n0001"]["issued_count"] == 1


async def test_checkpoint_restores_exact_request_facts_and_visible_import_marks_unknown():
    s = session(client=Client(ask=True))
    await s.respond("Make a card.")
    checkpoint = s.export_checkpoint()
    restored = session(client=Client(ask=True))
    restored.restore_checkpoint(checkpoint)
    assert restored.export_checkpoint() == checkpoint
    await restored.respond("Continue.")
    assert restored.interaction_state["n0001"]["issued_count"] == 1
    imported = session(client=Client(ask=True))
    imported.replace_history(s.history)
    await imported.respond("Continue.")
    assert not imported.interaction_state
    eligibility = next(
        x["context"]["request_eligibility"] for x in imported.client.calls if x["role"] == "intra"
    )
    assert eligibility["n0001"]["issued_count"] is None
    assert eligibility["n0001"]["reason"] == "history_unknown"
    broken = copy.deepcopy(checkpoint)
    broken["interaction"]["requests"] = {}
    with pytest.raises(ValueError):
        restored.restore_checkpoint(broken)


async def test_asking_plan_followed_by_generator_failure_never_counts_as_issuance():
    s = session(client=Client({"generator": ["bad"] * 4}, ask=True))
    with pytest.raises(RecoveryExhausted):
        await s.respond("Make a card.")
    assert not s.interaction_state and not s._events


def test_different_task_needs_keep_distinct_ids_when_applicability_changes():
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make two cards."}])
    raw = semantic(adjacent=False, needs=True)
    raw["goals"].append(goal("new:second"))
    raw["needs"].append(need("new:second_shape", "new:second"))
    snapshot, registry, _ = normalize_snapshot(
        SemanticSnapshot.model_validate(raw),
        IdentityIndex(),
        store,
        "t1",
        profile().goal_progression.policy,
    )
    assert snapshot.needs[0].ref != snapshot.needs[1].ref
    changed = snapshot.model_copy(deep=True)
    changed.needs[0].goal_id = snapshot.goals[1].ref
    result, candidate, _ = normalize_snapshot(
        changed, registry, store, "t2", profile().goal_progression.policy
    )
    assert [n.ref for n in result.needs] == [n.ref for n in snapshot.needs]
    assert result.needs[0].ref != result.needs[1].ref
    assert candidate.entries == registry.entries


def test_goal_scope_change_keeps_need_identity_and_old_request_fact():
    raw = semantic(needs=True)
    store = EvidenceStore([{"event_id": "m1", "role": "user", "content": "Make a card."}])
    snapshot, registry, _ = normalize_snapshot(
        SemanticSnapshot.model_validate(raw),
        IdentityIndex(),
        store,
        "t1",
        profile().goal_progression.policy,
    )
    original = snapshot.needs[0].ref
    snapshot.goals[0].scope, snapshot.goals[1].scope = "adjacent", "current"
    snapshot.goals[0].anchor_goal_ids = [snapshot.goals[1].ref]
    snapshot.goals[1].anchor_goal_ids = []
    result, _, _ = normalize_snapshot(
        snapshot, registry, store, "t2", profile().goal_progression.policy
    )
    assert result.needs[0].ref == original


async def test_failed_later_turn_does_not_overwrite_committed_turn():
    s = session(client=Client(ask=True))
    await s.respond("Make a card.")
    before = s.export_checkpoint()
    s.client.script["tracker"] = ["bad"] * 4
    with pytest.raises(RecoveryExhausted):
        await s.respond("Change it.")
    after = s.export_checkpoint()
    for key in ("visible_events", "snapshot", "identities", "interaction", "bridge", "version"):
        assert after[key] == before[key]
    s.reset()
    assert s.state is None and not s.history and s.interaction_state == {}


async def test_local_constraint_then_visible_reversion_preserves_material_and_identity():
    # The fixture supplies semantic judgments; this tests plumbing, not model inference quality.
    first = semantic(adjacent=False)
    first["constraints"] = [
        {
            "ref": "new:columns",
            "key": "columns",
            "value": "two",
            "applies_to": ["new:current"],
            "condition": "default layout",
            "basis": "explicit",
            "evidence_refs": [ref("m1")],
        }
    ]
    c = Client({"tracker": [first]})
    s = session(client=c)
    await s.respond("SOURCE CARD: A/B. Default to two columns.")

    def local(call):
        value = copy.deepcopy(call["context"]["previous_snapshot"])
        value["constraints"].append(
            {
                "ref": "new:exception",
                "key": "columns",
                "value": "one",
                "applies_to": ["g0001"],
                "condition": "only this export",
                "basis": "explicit",
                "evidence_refs": [ref("m3")],
            }
        )
        value["changed_goal_ids"] = ["g0001"]
        return value

    c.script["tracker"] = [local]
    await s.respond("For this export only, use one column.")
    assert len(s.state.constraints) == 2

    def revert(call):
        value = copy.deepcopy(call["context"]["previous_snapshot"])
        value["constraints"] = [value["constraints"][0]]
        return value

    c.script["tracker"] = [revert]
    await s.respond("Use the default again.")
    assert s.state.goals[0].ref == "g0001" and s.state.constraints[0].value == "two"
    assert len(s.state.constraints) == 1
    assert "SOURCE CARD: A/B" in str(c.calls[-1]["context"])
