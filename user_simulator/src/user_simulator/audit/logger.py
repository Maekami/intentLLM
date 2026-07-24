import json
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
        if enabled:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            with (self.run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    _sanitize(config_snapshot or {}, full=self.level == "full"),
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
            elif not full and lower in {"rendered_prompt", "messages", "raw_response"}:
                result[key] = "<hidden at summary audit level>"
            else:
                result[key] = _sanitize(item, full=full)
        return result
    if isinstance(value, list):
        return [_sanitize(item, full=full) for item in value]
    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump(mode="json"), full=full)
    return value
