from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from assistant.domain.messages import ChatMessage


class AssistantBaseline(ABC):
    name: str

    @abstractmethod
    def build_messages(self, history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Build the exact messages sent to the model."""


class BaseBaseline(AssistantBaseline):
    """Zero-shot baseline that adds no prompt or hidden instruction."""

    name = "base"

    def build_messages(self, history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        return [message.model_dump(mode="json", exclude_none=True) for message in history]
