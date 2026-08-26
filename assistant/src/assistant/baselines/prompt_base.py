from collections.abc import Sequence
from typing import Any

from assistant.baselines.base import AssistantBaseline
from assistant.domain.messages import ChatMessage
from assistant.prompt import SystemPrompt


class PromptBaseBaseline(AssistantBaseline):
    """Baseline that prepends the configured system prompt."""

    name = "prompt_base"

    def __init__(self, prompt: SystemPrompt) -> None:
        self.prompt = prompt

    def build_messages(self, history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.prompt.system},
            *[message.model_dump(mode="json", exclude_none=True) for message in history],
        ]
