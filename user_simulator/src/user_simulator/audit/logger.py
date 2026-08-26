import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import yaml

from user_simulator.audit.events import AuditEvent
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.state import EpisodeState


class AuditLogger:
    def __init__(
        self,
        sample_id: str,
        *,
        output_dir: str | Path = "runs",
        run_id: str | None = None,
        level: str = "full",
        enabled: bool = True,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        if level not in {"summary", "full"}:
            raise ValueError("audit level must be summary or full")
        self.sample_id = sample_id
        self.level = level
        self.enabled = enabled
        self.run_id = run_id or uuid.uuid4().hex
        self.run_dir = Path(output_dir) / self.run_id
        self.events: list[AuditEvent] = []
        self.git_commit = read_git_commit()
        if enabled:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            with (self.run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    _sanitize(
                        {**(config_snapshot or {}), "git_commit": self.git_commit},
                        full=self.level == "full",
                    ),
                    handle,
                    sort_keys=False,
                )

    def log(self, event_type: str, turn_index: int, payload: dict[str, Any]) -> None:
        sanitized = _sanitize(payload, full=self.level == "full")
        event = AuditEvent(
            event_type=event_type,
            sample_id=self.sample_id,
            turn_index=turn_index,
            payload=sanitized,
        )
        self.events.append(event)
        if self.enabled:
            with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")

    def log_message(self, message: ChatMessage, turn_index: int) -> None:
        if not self.enabled:
            return
        record = {"turn_index": turn_index, **message.model_dump()}
        with (self.run_dir / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_state(self, state: EpisodeState) -> None:
        if not self.enabled:
            return
        with (self.run_dir / "final_state.json").open("w", encoding="utf-8") as handle:
            handle.write(state.model_dump_json(indent=2))

    @property
    def latest_turn_events(self) -> list[AuditEvent]:
        if not self.events:
            return []
        turn = self.events[-1].turn_index
        return [event for event in self.events if event.turn_index == turn]

    def flush(self) -> None:
        """Files are synchronously flushed per append; retained for the CLI contract."""


def _sanitize(value: Any, *, full: bool) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lower = str(key).lower()
            if "api_key" in lower or lower == "authorization":
                result[key] = "<redacted>"
            elif not full and lower in {
                "messages",
                "rendered_prompt",
                "system",
                "system_prompt",
                "user_template",
            }:
                result[key] = "<hidden at summary audit level>"
            elif not full and lower in {
                "node_intent",
                "reason_text",
                "surface_user_message",
                "remaining_gap",
                "selected_remaining_gaps",
            }:
                result[key] = "<hidden latent node text>"
            elif not full and lower in {"raw_response", "invalid_raw_response"}:
                result[key] = _summarize_raw_response(item)
            else:
                result[key] = _sanitize(item, full=full)
        return result
    if isinstance(value, list):
        return [_sanitize(item, full=full) for item in value]
    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump(mode="json"), full=full)
    return value


def _summarize_raw_response(value: Any) -> Any:
    if not isinstance(value, dict):
        return "<hidden at summary audit level>"
    summarized: dict[str, Any] = {}
    if isinstance(value.get("decisions"), list):
        summarized["decisions"] = [
            {
                "node_id": item.get("node_id"),
                "exposable": item.get("exposable"),
            }
            for item in value["decisions"]
            if isinstance(item, dict)
        ]
    if "end_exposed" in value:
        summarized["end_exposed"] = value["end_exposed"]
    if isinstance(value.get("updates"), list):
        summarized["updates"] = [
            {
                "node_id": item.get("node_id"),
                "status": item.get("status"),
            }
            for item in value["updates"]
            if isinstance(item, dict)
        ]
    for key in (
        "selected_node_ids",
        "realization_mode",
        "contains_unsupported_task_content",
    ):
        if key in value:
            summarized[key] = value[key]
    if isinstance(value.get("coverage"), list):
        summarized["coverage"] = [
            {
                "node_id": item.get("node_id"),
                "covered": item.get("covered"),
            }
            for item in value["coverage"]
            if isinstance(item, dict)
        ]
    return _sanitize(summarized, full=False)


def read_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None
