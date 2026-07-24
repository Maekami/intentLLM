import pytest

from user_simulator.domain.enums import SatisfactionLevel
from user_simulator.domain.results import (
    CandidateExposureDecision,
    ControllerResult,
    NodeSatisfactionDecision,
    SatisfactionUpdateResult,
)
from user_simulator.engine.transitions import (
    apply_satisfaction_updates,
    normalize_controller_result,
    unresolved_queue,
)
from user_simulator.exceptions import ControllerOutputError, SatisfactionOutputError


def _decision(node: str, value: bool) -> CandidateExposureDecision:
    return CandidateExposureDecision(node_id=node, exposable=value, reason="audit")


def test_prefix_closure_forces_later_true_false() -> None:
    raw = ControllerResult(
        decisions=[_decision("N2", True), _decision("N3", False), _decision("N4", True)],
        end_reachable=True,
        summary="x",
    )
    result = normalize_controller_result(raw, ["N2", "N3", "N4"], True)
    assert result.newly_exposed == ["N2"]
    assert [x.exposable for x in result.decisions] == [True, False, False]
    assert not result.end_reachable
    assert result.violations


def test_end_requires_edge() -> None:
    raw = ControllerResult(decisions=[], end_reachable=True, summary="x")
    result = normalize_controller_result(raw, [], False)
    assert not result.end_reachable


def test_controller_candidates_must_match_exactly() -> None:
    raw = ControllerResult(decisions=[_decision("N2", True)], end_reachable=False, summary="x")
    with pytest.raises(ControllerOutputError):
        normalize_controller_result(raw, ["N2", "N3"], False)


def test_monotonic_satisfaction() -> None:
    raw = SatisfactionUpdateResult(
        updates=[
            NodeSatisfactionDecision(node_id="N1", status=SatisfactionLevel.UNSATISFIED, reason="x")
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
            NodeSatisfactionDecision(node_id="END", status=SatisfactionLevel.SATISFIED, reason="x")
        ],
        summary="x",
    )
    with pytest.raises(SatisfactionOutputError):
        apply_satisfaction_updates({"N1": SatisfactionLevel.UNSATISFIED}, raw, ["N1"])


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
