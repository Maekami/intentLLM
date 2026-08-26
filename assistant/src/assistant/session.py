from collections.abc import Sequence
from typing import Any

from assistant.baselines.base import AssistantBaseline
from assistant.config import GenerationSettings
from assistant.domain.messages import ChatMessage
from assistant.llm.base import ChatLLMClient


class AssistantSession:
    """A stateful multi-turn assistant backed by one baseline and one model."""

    def __init__(
        self,
        client: ChatLLMClient,
        generation: GenerationSettings,
        baseline: AssistantBaseline,
    ) -> None:
        self.client = client
        self.generation = generation
        self.baseline = baseline
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(self._history)

    @property
    def last_call_metadata(self) -> dict[str, Any]:
        metadata = getattr(self.client, "last_call_metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}

    async def respond(self, user_message: str) -> str:
        pending_user = ChatMessage(role="user", content=user_message)
        pending_history = [*self._history, pending_user]
        request_messages = self.baseline.build_messages(pending_history)
        generated = await self.client.generate(
            messages=request_messages,
            generation=self.generation,
        )
        assistant_message = ChatMessage(
            role="assistant",
            content=generated.content,
            reasoning_content=generated.reasoning_content,
            reasoning_details=generated.reasoning_details,
        )
        self._history.extend((pending_user, assistant_message))
        return generated.content

    def replace_history(self, messages: Sequence[ChatMessage]) -> None:
        self._history = list(messages)

    def finalize_task(
        self,
        *,
        task_id: str | None = None,
        success: bool,
        feedback: str | None = None,
    ) -> dict[str, Any] | None:
        """Finish one externally evaluated task; memory-free sessions do nothing."""

        return None

    def reset(self) -> None:
        self._history.clear()
