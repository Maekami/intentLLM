from user_simulator.data.validator import node_index
from user_simulator.domain.enums import SatisfactionLevel
from user_simulator.domain.results import (
    CandidateExposureDecision,
    ControllerResult,
    NormalizedControllerResult,
    SatisfactionUpdateResult,
)
from user_simulator.exceptions import ControllerOutputError, SatisfactionOutputError

_RANK = {
    SatisfactionLevel.UNSATISFIED: 0,
    SatisfactionLevel.PARTIALLY_SATISFIED: 1,
    SatisfactionLevel.SATISFIED: 2,
}


def normalize_controller_result(
    raw: ControllerResult,
    candidates: list[str],
    has_end_edge: bool,
) -> NormalizedControllerResult:
    received = [item.node_id for item in raw.decisions]
    if len(received) != len(set(received)):
        raise ControllerOutputError("controller returned duplicate candidate decisions")
    if set(received) != set(candidates) or len(received) != len(candidates):
        raise ControllerOutputError(
            f"controller decisions must exactly match candidates {candidates}; got {received}"
        )
    by_id = {item.node_id: item for item in raw.decisions}
    normalized: list[CandidateExposureDecision] = []
    violations: list[str] = []
    prefix_open = True
    newly_exposed: list[str] = []
    for node_id in candidates:
        item = by_id[node_id]
        exposable = item.exposable and prefix_open
        if item.exposable and not prefix_open:
            violations.append(f"forced {node_id}=false after earlier false decision")
        if not item.exposable:
            prefix_open = False
        if exposable:
            newly_exposed.append(node_id)
        normalized.append(item.model_copy(update={"exposable": exposable}))

    all_candidates_exposable = len(newly_exposed) == len(candidates)
    end_reachable = raw.end_reachable and has_end_edge and all_candidates_exposable
    if raw.end_reachable and not has_end_edge:
        violations.append("forced END=false because no outgoing END edge exists")
    elif raw.end_reachable and not all_candidates_exposable:
        violations.append("forced END=false because candidate prefix is incomplete")
    return NormalizedControllerResult(
        decisions=normalized,
        newly_exposed=newly_exposed,
        end_reachable=end_reachable,
        violations=violations,
        summary=raw.summary,
    )


def apply_satisfaction_updates(
    previous: dict[str, SatisfactionLevel],
    raw: SatisfactionUpdateResult,
    exposed_nodes: list[str],
    *,
    monotonic: bool = True,
) -> tuple[dict[str, SatisfactionLevel], list[str]]:
    received = [item.node_id for item in raw.updates]
    if "END" in received:
        raise SatisfactionOutputError("satisfaction output must not include END")
    if len(received) != len(set(received)):
        raise SatisfactionOutputError("duplicate satisfaction decisions")
    if set(received) != set(exposed_nodes) or len(received) != len(exposed_nodes):
        raise SatisfactionOutputError(
            f"updates must exactly match exposed nodes {exposed_nodes}; got {received}"
        )
    by_id = {item.node_id: item.status for item in raw.updates}
    applied: dict[str, SatisfactionLevel] = {}
    violations: list[str] = []
    for node_id in exposed_nodes:
        old = previous[node_id]
        proposed = by_id[node_id]
        if monotonic and _RANK[proposed] < _RANK[old]:
            applied[node_id] = old
            violations.append(
                f"prevented satisfaction regression for {node_id}: {old.value}->{proposed.value}"
            )
        else:
            applied[node_id] = proposed
    return applied, violations


def unresolved_queue(
    exposed_nodes: list[str], satisfaction: dict[str, SatisfactionLevel]
) -> list[str]:
    return sorted(
        (
            node_id
            for node_id in exposed_nodes
            if satisfaction[node_id] != SatisfactionLevel.SATISFIED
        ),
        key=node_index,
    )


def all_exposed_satisfied(
    exposed_nodes: list[str], satisfaction: dict[str, SatisfactionLevel]
) -> bool:
    return all(satisfaction[node_id] == SatisfactionLevel.SATISFIED for node_id in exposed_nodes)
