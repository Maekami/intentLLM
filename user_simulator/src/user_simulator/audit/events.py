from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    sample_id: str
    turn_index: int
    payload: dict[str, Any] = Field(default_factory=dict)
