from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class MemoryEntry:
    """Structured task experience m_i = S(x_i, y_i, f_i)."""

    task_id: str
    input_text: str
    output_text: str
    feedback: str | None = None
    trajectory: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    is_successful: bool = False

    def __post_init__(self) -> None:
        if not self.task_id:
            value = f"{self.input_text}\0{self.output_text}".encode()
            self.task_id = hashlib.sha256(value).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "feedback": self.feedback,
            "trajectory": self.trajectory,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "is_successful": self.is_successful,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MemoryEntry:
        if not isinstance(value, dict):
            raise TypeError("memory entry must be a mapping")
        required = ("task_id", "input_text", "output_text")
        missing = [key for key in required if not isinstance(value.get(key), str)]
        if missing:
            raise ValueError(f"memory entry has invalid fields: {', '.join(missing)}")
        trajectory = value.get("trajectory") or []
        metadata = value.get("metadata") or {}
        if not isinstance(trajectory, list) or not all(
            isinstance(item, dict) for item in trajectory
        ):
            raise ValueError("memory entry trajectory must be a list of mappings")
        if not isinstance(metadata, dict):
            raise TypeError("memory entry metadata must be a mapping")
        feedback = value.get("feedback")
        if feedback is not None and not isinstance(feedback, str):
            raise ValueError("memory entry feedback must be text or null")
        timestamp = value.get("timestamp")
        if not isinstance(timestamp, str):
            timestamp = datetime.now(UTC).isoformat()
        return cls(
            task_id=value["task_id"],
            input_text=value["input_text"],
            output_text=value["output_text"],
            feedback=feedback,
            trajectory=[
                {str(key): str(item_value) for key, item_value in item.items()}
                for item in trajectory
            ],
            metadata={str(key): item for key, item in metadata.items()},
            timestamp=timestamp,
            is_successful=value.get("is_successful") is True,
        )

    def to_text(
        self,
        *,
        include_trajectory: bool = True,
        include_feedback: bool = True,
    ) -> str:
        parts = [f"Task: {self.input_text}"]
        if include_trajectory and self.trajectory:
            lines: list[str] = []
            for step in self.trajectory:
                if user := step.get("user"):
                    lines.append(f"  User: {user}")
                if assistant := step.get("assistant"):
                    lines.append(f"  Assistant: {assistant}")
                if action := step.get("action"):
                    lines.append(f"  Action: {action}")
                if observation := step.get("observation"):
                    lines.append(f"  Observation: {observation}")
            if lines:
                parts.append("Trajectory:\n" + "\n".join(lines))
        parts.append(f"Output: {self.output_text}")
        if include_feedback and self.feedback:
            parts.append(f"Feedback: {self.feedback}")
        parts.append(f"Result: {'Success' if self.is_successful else 'Failure'}")
        return "\n".join(parts)


@dataclass(frozen=True)
class RetrievalResult:
    entry: MemoryEntry
    score: float
    rank: int


@dataclass(frozen=True)
class MemoryUpdateResult:
    stored: bool
    task_id: str | None
    path: str
    entry_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stored": self.stored,
            "task_id": self.task_id,
            "path": self.path,
            "entry_count": self.entry_count,
            "reason": self.reason,
        }
