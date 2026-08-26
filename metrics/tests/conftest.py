from __future__ import annotations

from pathlib import Path

from intent_metrics.models import ConversationMessage, EpisodeTrace


def make_trace(
    *,
    events: list[dict],
    turns: int,
    intent_node_ids: tuple[str, ...] = ("N1", "N2"),
    initial_exposed: tuple[str, ...] = ("N1",),
    run_dir: Path = Path("/tmp/metric-fixture"),
) -> EpisodeTrace:
    conversation = [ConversationMessage(0, "user", "Please help me.")]
    for turn in range(1, turns + 1):
        conversation.append(ConversationMessage(turn, "assistant", f"Answer {turn}"))
        conversation.append(ConversationMessage(turn, "user", f"Follow-up {turn}"))
    return EpisodeTrace(
        run_dir=run_dir,
        run_id=run_dir.name,
        sample_id="fixture",
        difficulty="hard",
        assistant_model="fake/model",
        intent_node_ids=intent_node_ids,
        initial_exposed_node_ids=initial_exposed,
        events=tuple(events),
        conversation=tuple(conversation),
        source_outcome="completed" if turns < 20 else "budget_exhausted",
    )


def turn_events(
    turn: int,
    *,
    after: dict[str, str],
    newly_exposed: list[str] | None = None,
    output_tokens: int = 10,
) -> list[dict]:
    exposed = list(after)
    return [
        {
            "event_type": "assistant_generation_completed",
            "turn_index": turn,
            "payload": {"llm_call": {"output_tokens": output_tokens}},
        },
        {"event_type": "assistant_message_received", "turn_index": turn, "payload": {}},
        {
            "event_type": "nodes_exposed",
            "turn_index": turn,
            "payload": {"newly_exposed": newly_exposed or []},
        },
        {
            "event_type": "satisfaction_requested",
            "turn_index": turn,
            "payload": {"exposed_nodes": exposed},
        },
        {
            "event_type": "satisfaction_applied",
            "turn_index": turn,
            "payload": {"after": after},
        },
    ]
