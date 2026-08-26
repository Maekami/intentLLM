import pytest
from pydantic import ValidationError

from user_simulator.domain.enums import SatisfactionLevel
from user_simulator.domain.results import (
    CandidateExposureDecision,
    ControllerResult,
    NodeSatisfactionDecision,
    SatisfactionUpdateResult,
    make_node_satisfaction_decision,
)
from user_simulator.engine.transitions import (
    apply_satisfaction_updates,
    derive_system_end_exposure,
    normalize_controller_result,
    unresolved_queue,
)
from user_simulator.exceptions import ControllerOutputError, SatisfactionOutputError


def _decision(node: str, value: bool) -> CandidateExposureDecision:
    return CandidateExposureDecision(node_id=node, exposable=value, reason="audit")


def test_prefix_closure_forces_later_true_false() -> None:
    raw = ControllerResult(
        decisions=[_decision("N2", True), _decision("N3", False), _decision("N4", True)],
        summary="x",
    )
    result = normalize_controller_result(raw, ["N2", "N3", "N4"])
    assert result.newly_exposed == ["N2"]
    assert [x.exposable for x in result.decisions] == [True, False, False]
    assert result.violations


def test_system_exposes_end_at_terminal_current_frontier() -> None:
    rule = derive_system_end_exposure(
        candidate_ids=[],
        newly_exposed=[],
        has_end_edge_before=True,
        outgoing_intents_after=[],
        has_end_edge_after=True,
    )
    assert rule == "outgoing_end_after_complete_prefix"


def test_system_does_not_expose_end_for_incomplete_intent_prefix() -> None:
    rule = derive_system_end_exposure(
        candidate_ids=["N2", "N3", "N4"],
        newly_exposed=["N2"],
        has_end_edge_before=True,
        outgoing_intents_after=["N3", "N4"],
        has_end_edge_after=False,
    )
    assert rule is None


def test_system_closes_terminal_only_frontier_reached_by_multi_node_step() -> None:
    rule = derive_system_end_exposure(
        candidate_ids=["N3", "N4"],
        newly_exposed=["N3", "N4"],
        has_end_edge_before=False,
        outgoing_intents_after=[],
        has_end_edge_after=True,
    )
    assert rule == "terminal_only_frontier_after_advance"


def test_system_does_not_close_advanced_frontier_with_unexposed_intents() -> None:
    rule = derive_system_end_exposure(
        candidate_ids=["N2"],
        newly_exposed=["N2"],
        has_end_edge_before=False,
        outgoing_intents_after=["N3"],
        has_end_edge_after=True,
    )
    assert rule is None


def test_controller_candidates_must_match_exactly() -> None:
    raw = ControllerResult(decisions=[_decision("N2", True)], summary="x")
    with pytest.raises(ControllerOutputError):
        normalize_controller_result(raw, ["N2", "N3"])


def test_monotonic_satisfaction() -> None:
    raw = SatisfactionUpdateResult(
        updates=[
            make_node_satisfaction_decision(
                node_id="N1",
                status=SatisfactionLevel.UNSATISFIED,
                reason="x",
                remaining_gap=None,
            )
        ],
        summary="x",
    )
    applied, violations = apply_satisfaction_updates(
        {"N1": SatisfactionLevel.SATISFIED}, raw, ["N1"]
    )
    assert applied["N1"] == SatisfactionLevel.SATISFIED
    assert violations


def test_satisfaction_requires_exact_set_and_excludes_end() -> None:
    raw = SatisfactionUpdateResult(
        updates=[
            make_node_satisfaction_decision(
                node_id="END",
                status=SatisfactionLevel.SATISFIED,
                reason="x",
                remaining_gap=None,
            )
        ],
        summary="x",
    )
    with pytest.raises(SatisfactionOutputError):
        apply_satisfaction_updates({"N1": SatisfactionLevel.UNSATISFIED}, raw, ["N1"])


@pytest.mark.parametrize(
    ("status", "remaining_gap"),
    [
        (SatisfactionLevel.PARTIALLY_SATISFIED, None),
        (SatisfactionLevel.UNSATISFIED, "A gap must not be supplied here."),
        (SatisfactionLevel.SATISFIED, "A gap must not be supplied here."),
    ],
)
def test_remaining_gap_matches_satisfaction_status(status, remaining_gap) -> None:
    with pytest.raises(ValidationError):
        NodeSatisfactionDecision(
            node_id="N1",
            status=status,
            reason="The status is supported by concrete assistant evidence.",
            remaining_gap=remaining_gap,
        )


def test_partial_satisfaction_accepts_a_remaining_gap() -> None:
    decision = NodeSatisfactionDecision(
        node_id="N1",
        status=SatisfactionLevel.PARTIALLY_SATISFIED,
        reason="The assistant addressed setup but did not explain verification.",
        remaining_gap="Explain how the user can verify that setup succeeded.",
    )
    assert decision.remaining_gap == "Explain how the user can verify that setup succeeded."


def test_unresolved_contains_partial_and_is_sorted() -> None:
    queue = unresolved_queue(
        ["N3", "N1", "N2"],
        {
            "N1": SatisfactionLevel.SATISFIED,
            "N2": SatisfactionLevel.PARTIALLY_SATISFIED,
            "N3": SatisfactionLevel.UNSATISFIED,
        },
    )
    assert queue == ["N2", "N3"]
