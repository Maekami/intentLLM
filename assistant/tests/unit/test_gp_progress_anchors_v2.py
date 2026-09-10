"""Progress can start from addressed work without falsely reopening it."""

import pytest
from gp_helpers import goal, profile, ref, semantic

from assistant.goal_progression.context import ContextBuilder
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.events import EvidenceStore, IdentityIndex, InteractionIndex
from assistant.goal_progression.schemas import Goal, SemanticSnapshot
from assistant.goal_progression.tracker import normalize_snapshot


def case(scope="current", status="addressed"):
    raw = semantic()
    raw["goals"][0].update(
        scope=scope,
        status=status,
        remaining_work="" if status == "addressed" else "Prepare the card.",
        anchor_goal_ids=["new:remaining_current"]
        if scope == "adjacent" and status == "active"
        else [],
        evidence_refs=[ref("m1"), ref("m2")],
    )
    raw["goals"].append(goal("new:remaining_current", event="m3"))
    raw["goals"][-1]["description"] = "Compare two spacing choices for the label."
    store = EvidenceStore(
        [
            {"event_id": "m1", "role": "user", "content": "Prepare a card and matching label."},
            {"event_id": "m2", "role": "assistant", "content": "The card is complete."},
            {
                "event_id": "m3",
                "role": "user",
                "content": "Now compare the label's spacing choices.",
            },
        ]
    )
    return SemanticSnapshot.model_validate(raw), store


@pytest.mark.parametrize("scope", ["current", "adjacent"])
def test_addressed_result_is_a_valid_progress_anchor_without_reactivation(scope):
    raw, store = case(scope)
    snapshot, _, _ = normalize_snapshot(
        raw, IdentityIndex(), store, "t2", profile().goal_progression.policy
    )
    assert snapshot.goals[0].status == "addressed"
    assert snapshot.goals[1].anchor_goal_ids == [snapshot.goals[0].ref]


@pytest.mark.parametrize(
    "scope,status", [("current", "paused"), ("adjacent", "paused"), ("adjacent", "active")]
)
def test_paused_or_unfulfilled_adjacent_goal_is_not_an_independent_anchor(scope, status):
    raw, store = case(scope, status)
    with pytest.raises(ContractError) as error:
        normalize_snapshot(raw, IdentityIndex(), store, "t2", profile().goal_progression.policy)
    assert "Allowed refs" in error.value.detail
    assert "new:remaining_current" in error.value.detail
    assert "anchor_goal_ids" in error.value.field_path


def test_planners_receive_only_relevant_anchor_summaries_not_archived_goal_details():
    raw, store = case()
    unrelated = goal("new:archive", event="m1")
    unrelated.update(status="addressed", description="UNRELATED ARCHIVE", remaining_work="")
    raw.goals.append(Goal.model_validate(unrelated))
    snapshot, registry, _ = normalize_snapshot(
        raw, IdentityIndex(), store, "t2", profile().goal_progression.policy
    )
    builder = ContextBuilder(store, registry, InteractionIndex())
    for view in (builder.planner(snapshot, "inter"), builder.joint(snapshot)):
        anchors = view["progress_anchors"]
        assert {a["ref"] for a in anchors} == {snapshot.goals[0].ref, snapshot.goals[2].ref}
        assert any(a["status"] == "addressed" for a in anchors)
        assert all(a["description"] != "UNRELATED ARCHIVE" for a in anchors)
    assert "progress_anchors" not in builder.planner(snapshot, "intra")
