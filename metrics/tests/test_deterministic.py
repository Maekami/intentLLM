from __future__ import annotations

from conftest import make_trace, turn_events

from intent_metrics.deterministic import (
    FAILURE_TURN,
    compute_assistant_tokens,
    compute_dag_turn_metrics,
)


def test_exposure_at_turn_three_and_satisfaction_at_turn_five() -> None:
    events = []
    states = [
        {"N1": "unsatisfied"},
        {"N1": "partially_satisfied"},
        {"N1": "satisfied", "N2": "partially_satisfied"},
        {"N1": "satisfied", "N2": "partially_satisfied"},
        {"N1": "satisfied", "N2": "satisfied"},
    ]
    for turn, state in enumerate(states, 1):
        events.extend(
            turn_events(
                turn,
                after=state,
                newly_exposed=["N2"] if turn == 3 else None,
            )
        )
    metric = compute_dag_turn_metrics(make_trace(events=events, turns=5))
    assert metric.all_node_exposure_turn == 3
    assert metric.all_node_satisfaction_turn == 5


def test_partial_satisfaction_does_not_count_as_satisfied() -> None:
    events = []
    for turn in range(1, 8):
        state = {
            "N1": "satisfied",
            "N2": "satisfied" if turn == 7 else "partially_satisfied",
        }
        events.extend(
            turn_events(
                turn,
                after=state,
                newly_exposed=["N2"] if turn == 1 else None,
            )
        )
    metric = compute_dag_turn_metrics(make_trace(events=events, turns=7))
    assert metric.all_node_exposure_turn == 1
    assert metric.all_node_satisfaction_turn == 7


def test_completion_failure_is_twenty_one() -> None:
    events = []
    for turn in range(1, 21):
        events.extend(turn_events(turn, after={"N1": "satisfied"}))
    metric = compute_dag_turn_metrics(make_trace(events=events, turns=20))
    assert metric.all_node_exposure_turn == FAILURE_TURN
    assert metric.all_node_satisfaction_turn == FAILURE_TURN


def test_all_nodes_first_exposed_exactly_at_turn_twenty() -> None:
    events = []
    for turn in range(1, 21):
        state = {"N1": "satisfied"}
        if turn == 20:
            state["N2"] = "unsatisfied"
        events.extend(
            turn_events(
                turn,
                after=state,
                newly_exposed=["N2"] if turn == 20 else None,
            )
        )
    metric = compute_dag_turn_metrics(make_trace(events=events, turns=20))
    assert metric.all_node_exposure_turn == 20
    assert metric.all_node_satisfaction_turn == FAILURE_TURN


def test_auto_exposed_backbone_node_belongs_to_current_turn() -> None:
    events = turn_events(1, after={"N1": "satisfied"})
    events.append(
        {
            "event_type": "backbone_node_auto_exposed",
            "turn_index": 1,
            "payload": {"node_id": "N2"},
        }
    )
    events.extend(turn_events(2, after={"N1": "satisfied", "N2": "satisfied"}))
    metric = compute_dag_turn_metrics(make_trace(events=events, turns=2))
    assert metric.all_node_exposure_turn == 1
    assert metric.all_node_satisfaction_turn == 2


def test_single_initial_node_has_zero_exposure_turn() -> None:
    events = turn_events(1, after={"N1": "satisfied"})
    trace = make_trace(
        events=events,
        turns=1,
        intent_node_ids=("N1",),
        initial_exposed=("N1",),
    )
    metric = compute_dag_turn_metrics(trace)
    assert metric.all_node_exposure_turn == 0
    assert metric.all_node_satisfaction_turn == 1


def test_tokens_are_summed_per_episode_and_unaccepted_generation_is_excluded() -> None:
    events = []
    for turn, tokens in enumerate((120, 180, 100), 1):
        events.extend(
            turn_events(
                turn,
                after={"N1": "satisfied", "N2": "satisfied"},
                newly_exposed=["N2"] if turn == 1 else None,
                output_tokens=tokens,
            )
        )
    events.append(
        {
            "event_type": "assistant_generation_completed",
            "turn_index": 4,
            "payload": {"llm_call": {"output_tokens": 999}},
        }
    )
    result = compute_assistant_tokens(make_trace(events=events, turns=3))
    assert result.assistant_tokens == 400


def test_tokens_fall_back_to_answer_plus_thinking() -> None:
    events = turn_events(1, after={"N1": "satisfied"})
    completed = next(
        item for item in events if item["event_type"] == "assistant_generation_completed"
    )
    completed["payload"]["llm_call"] = {"answer_tokens": 80, "thinking_tokens": 20}
    trace = make_trace(
        events=events,
        turns=1,
        intent_node_ids=("N1",),
        initial_exposed=("N1",),
    )
    result = compute_assistant_tokens(trace)
    assert result.assistant_tokens == 100
    assert result.fallback_turns == (1,)


def test_delivered_text_tokens_ignore_internal_calls_and_sum_turns():
    events = []
    for turn, tokens, calls in [(1, 7, 4), (2, 11, 9)]:
        current = turn_events(turn, after={'N1': 'satisfied'}, output_tokens=9999)
        current[0]['payload']['llm_call'].update(
            visible_response_tokens=tokens, internal_call_count=calls,
            goal_progression={'calls': [{'output_tokens': 10000}] * calls},
        )
        events.extend(current)
    assert compute_assistant_tokens(make_trace(events=events, turns=2)).assistant_tokens == 18


def test_unknown_visible_tokens_never_fall_back_to_internal_usage():
    import pytest
    from intent_metrics.errors import MetricDataError
    events = turn_events(1, after={'N1': 'satisfied'}, output_tokens=9999)
    events[0]['payload']['llm_call']['visible_response_tokens'] = None
    with pytest.raises(MetricDataError, match='visible tokens unknown'):
        compute_assistant_tokens(make_trace(events=events, turns=1))
