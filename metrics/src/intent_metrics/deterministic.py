from __future__ import annotations

from collections import defaultdict
from typing import Any

from intent_metrics.errors import MetricDataError
from intent_metrics.models import DagTurnMetrics, EpisodeTrace, TokenMetric

T_MAX = 20
FAILURE_TURN = T_MAX + 1
_SATISFIED = "satisfied"
_UNSATISFIED = "unsatisfied"
_SATISFACTION_VALUES = {_UNSATISFIED, "partially_satisfied", _SATISFIED}


def compute_dag_turn_metrics(trace: EpisodeTrace) -> DagTurnMetrics:
    """Compute exposure/satisfaction completion turns from simulator state events."""

    required = set(trace.intent_node_ids)
    if not required:
        raise MetricDataError("an episode must contain at least one intent node")
    exposed = set(trace.initial_exposed_node_ids)
    unknown_initial = exposed - required
    if unknown_initial:
        raise MetricDataError(
            f"initial exposure contains non-intent nodes: {sorted(unknown_initial)}"
        )

    accepted_turns = _accepted_turns(trace)
    events_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in trace.events:
        turn = event.get("turn_index")
        if isinstance(turn, int) and not isinstance(turn, bool) and turn in accepted_turns:
            events_by_turn[turn].append(event)

    exposure_turn = 0 if required.issubset(exposed) else FAILURE_TURN
    satisfaction_turn = FAILURE_TURN
    satisfaction: dict[str, str] = {}

    for turn in sorted(accepted_turns):
        saw_satisfaction_snapshot = False
        for event in events_by_turn[turn]:
            event_type = event.get("event_type")
            payload = event.get("payload")
            if not isinstance(payload, dict):
                payload = {}

            if event_type == "nodes_exposed":
                exposed.update(_node_list(payload.get("newly_exposed"), event_type, required))
            elif event_type == "satisfaction_requested":
                exposed.update(_node_list(payload.get("exposed_nodes"), event_type, required))
            elif event_type == "satisfaction_applied":
                satisfaction = _satisfaction_snapshot(payload.get("after"), required)
                exposed.update(satisfaction)
                saw_satisfaction_snapshot = True
            elif event_type == "backbone_node_auto_exposed":
                node_id = payload.get("node_id")
                if not isinstance(node_id, str) or node_id not in required:
                    raise MetricDataError(
                        f"turn {turn}: backbone_node_auto_exposed has invalid node_id {node_id!r}"
                    )
                exposed.add(node_id)
                satisfaction[node_id] = _UNSATISFIED

        if not saw_satisfaction_snapshot:
            raise MetricDataError(f"turn {turn}: canonical satisfaction_applied event is missing")
        if exposure_turn == FAILURE_TURN and required.issubset(exposed):
            exposure_turn = turn
        if satisfaction_turn == FAILURE_TURN and all(
            satisfaction.get(node_id) == _SATISFIED for node_id in required
        ):
            satisfaction_turn = turn

    warnings: list[str] = []
    if exposure_turn > satisfaction_turn:
        warnings.append(
            "unexpected invariant violation: all-node satisfaction precedes all-node exposure "
            f"({satisfaction_turn} < {exposure_turn})"
        )
    return DagTurnMetrics(
        all_node_exposure_turn=exposure_turn,
        all_node_satisfaction_turn=satisfaction_turn,
        warnings=tuple(warnings),
    )


def compute_assistant_tokens(trace: EpisodeTrace) -> TokenMetric:
    """Sum output tokens for assistant responses accepted as evaluation turns."""

    accepted_turns = _accepted_turns(trace)
    completed: dict[int, dict[str, Any]] = {}
    for event in trace.events:
        if event.get("event_type") != "assistant_generation_completed":
            continue
        turn = event.get("turn_index")
        if turn not in accepted_turns:
            # The current pipeline may generate turn T_MAX+1 before the simulator
            # rejects it. It is not a submitted assistant interaction turn.
            continue
        if turn in completed:
            raise MetricDataError(f"turn {turn}: duplicate assistant_generation_completed event")
        completed[turn] = event

    missing = sorted(accepted_turns - set(completed))
    if missing:
        raise MetricDataError(f"assistant token metadata is missing for accepted turns {missing}")

    total = 0
    fallback_turns: list[int] = []
    for turn in sorted(accepted_turns):
        payload = completed[turn].get("payload")
        llm_call = payload.get("llm_call") if isinstance(payload, dict) else None
        if not isinstance(llm_call, dict):
            raise MetricDataError(f"turn {turn}: assistant llm_call metadata is missing")
        output_tokens = _nonnegative_int(llm_call.get("output_tokens"))
        if output_tokens is not None:
            total += output_tokens
            continue

        answer_tokens = _nonnegative_int(llm_call.get("answer_tokens"))
        thinking_tokens = _nonnegative_int(llm_call.get("thinking_tokens"))
        if answer_tokens is not None and thinking_tokens is not None:
            total += answer_tokens + thinking_tokens
            fallback_turns.append(turn)
            continue
        if answer_tokens is not None:
            # Some providers expose only visible completion tokens. Preserve the
            # available project-standard count and make the limitation explicit.
            total += answer_tokens
            fallback_turns.append(turn)
            continue
        raise MetricDataError(f"turn {turn}: no usable assistant output-token count is available")
    return TokenMetric(assistant_tokens=total, fallback_turns=tuple(fallback_turns))


def _accepted_turns(trace: EpisodeTrace) -> set[int]:
    event_turns: set[int] = set()
    for event in trace.events:
        if event.get("event_type") != "assistant_message_received":
            continue
        turn = event.get("turn_index")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
            raise MetricDataError(f"assistant_message_received has invalid turn_index {turn!r}")
        if turn <= T_MAX:
            event_turns.add(turn)
    if not event_turns:
        raise MetricDataError("trace contains no accepted assistant interaction turn")
    expected = set(range(1, max(event_turns) + 1))
    if event_turns != expected:
        raise MetricDataError(f"accepted assistant turns are not contiguous: {sorted(event_turns)}")

    transcript_turns = {
        message.turn_index for message in trace.conversation if message.role == "assistant"
    }
    if transcript_turns != event_turns:
        raise MetricDataError(
            "accepted assistant turns disagree between events and transcript: "
            f"events={sorted(event_turns)}, transcript={sorted(transcript_turns)}"
        )
    return event_turns


def _node_list(value: Any, event_type: str, required: set[str]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MetricDataError(f"{event_type} has no valid node list")
    unknown = set(value) - required
    if unknown:
        raise MetricDataError(f"{event_type} contains non-intent nodes: {sorted(unknown)}")
    return value


def _satisfaction_snapshot(value: Any, required: set[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        raise MetricDataError("satisfaction_applied.after must be a mapping")
    result: dict[str, str] = {}
    for node_id, status in value.items():
        if not isinstance(node_id, str) or node_id not in required:
            raise MetricDataError(
                f"satisfaction_applied.after contains non-intent node {node_id!r}"
            )
        if status not in _SATISFACTION_VALUES:
            raise MetricDataError(
                f"satisfaction_applied.after has invalid status {status!r} for {node_id}"
            )
        result[node_id] = status
    return result


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
