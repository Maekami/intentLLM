from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from user_simulator.audit.logger import AuditLogger
from user_simulator.domain.state import EpisodeState


class ConsoleAuditRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def latest(self, logger: AuditLogger) -> None:
        events = {event.event_type: event.payload for event in logger.latest_turn_events}
        if not events:
            self.console.print(Panel("No audit events yet.", title="Turn audit"))
            return
        turn = logger.latest_turn_events[-1].turn_index
        controller_request = events.get("controller_requested", {})
        raw_controller = events.get("controller_raw_result", {})
        normalized = events.get("controller_prefix_normalized", {})
        exposure = events.get("nodes_exposed", {})
        satisfaction_raw = events.get("satisfaction_raw_result", {})
        satisfaction = events.get("satisfaction_applied", {})
        termination = events.get("termination_checked", {})
        selection = events.get("nodes_selected", {})
        generation_request = events.get("user_generation_requested", {})
        generated = events.get("user_message_generated", {})
        auto = events.get("backbone_node_auto_exposed", {})
        raw_controller_response = _as_dict(raw_controller.get("raw_response"))

        sections = [
            self._section(
                "TURN",
                {
                    "turn index": turn,
                    "sample ID": logger.sample_id,
                },
            ),
            self._section(
                "CONTROLLER",
                {
                    "frontier before": controller_request.get("frontier"),
                    "outgoing candidates": controller_request.get("candidates"),
                    "raw decisions": raw_controller_response.get("decisions")
                    or "<hidden at summary audit level>",
                    "normalized decisions": normalized.get("decisions"),
                    "prefix violations": normalized.get("violations"),
                    "newly exposed": exposure.get("newly_exposed"),
                    "END reachable": exposure.get("end_reachable"),
                    "frontier after": exposure.get("frontier_after"),
                },
            ),
            self._section(
                "SATISFACTION",
                {
                    "status before": satisfaction.get("before"),
                    "proposed status": satisfaction.get("proposed")
                    or _status_map(satisfaction_raw),
                    "applied status": satisfaction.get("after"),
                    "monotonicity violations": satisfaction.get("violations"),
                },
            ),
            self._section(
                "SYSTEM",
                {
                    "all exposed satisfied": termination.get("all_exposed_satisfied"),
                    "termination result": termination.get("will_terminate"),
                    "auto-exposed backbone": auto.get("node_id"),
                },
            ),
            self._section(
                "USER POLICY",
                {
                    "unresolved queue": events.get("unresolved_queue_built", {}).get("queue"),
                    "selected nodes": selection.get("selected_nodes"),
                    "selection rule": selection.get("selection_rule"),
                    "realization mode": generation_request.get("mode"),
                    "generated message": generated.get("user_message"),
                },
            ),
            self._llm_section(
                [
                    raw_controller.get("llm_call", {}),
                    satisfaction_raw.get("llm_call", {}),
                    events.get("user_generation_raw_result", {}).get("llm_call", {}),
                ]
            ),
        ]
        self.console.print(Panel(Group(*sections), title=f"Consolidated audit · turn {turn}"))

    def raw_latest(self, logger: AuditLogger) -> None:
        payload = [
            {"event": event.event_type, **event.payload} for event in logger.latest_turn_events
        ]
        self.console.print(Panel(Pretty(payload), title="Raw latest-turn audit"))

    def state(self, state: EpisodeState) -> None:
        self.console.print(Panel(Pretty(state.model_dump(mode="json")), title="State"))

    @staticmethod
    def _section(title: str, values: dict[str, Any]) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold")
        table.add_column()
        for label, value in values.items():
            table.add_row(label, Pretty(value))
        return Panel(table, title=title, border_style="dim")

    def _llm_section(self, calls: list[dict[str, Any]]) -> Panel:
        rows: dict[str, Any] = {}
        for call in calls:
            if not call:
                continue
            component = call.get("component", "unknown")
            rows[component] = {
                "model": call.get("model_id"),
                "schema": _versioned_hash(call, "schema"),
                "prompt": _versioned_hash(call, "prompt"),
                "latency": call.get("latency_seconds"),
                "tokens": {
                    "input": call.get("input_tokens"),
                    "output": call.get("output_tokens"),
                },
                "retries": {
                    "transport": call.get("transport_retry_count"),
                    "semantic": call.get("semantic_retry_count"),
                },
                "validation": {
                    "structured": call.get("structured_validation_status"),
                    "semantic": call.get("semantic_validation_status"),
                },
            }
        return self._section("LLM", rows)


def _status_map(raw_event: dict[str, Any]) -> dict[str, str] | None:
    updates = _as_dict(raw_event.get("raw_response")).get("updates")
    if not updates:
        return None
    return {item["node_id"]: item["status"] for item in updates}


def _versioned_hash(call: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "name": call.get(f"{prefix}_name"),
        "version": call.get(f"{prefix}_version"),
        "hash": call.get(f"{prefix}_hash"),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
