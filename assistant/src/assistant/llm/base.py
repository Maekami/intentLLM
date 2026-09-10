from dataclasses import dataclass
from typing import Any, Protocol

from assistant.config import GenerationSettings


@dataclass(frozen=True)
class GeneratedResponse:
    content: str
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    # Per-response metadata avoids races when one client serves concurrent
    # offline distillation calls. Sessions retain last_call_metadata for
    # backward compatibility.
    metadata: dict[str, Any] | None = None


class ChatLLMClient(Protocol):
    last_call_metadata: dict[str, Any]

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        generation: GenerationSettings,
        response_schema: dict[str, Any] | None = None,
    ) -> GeneratedResponse: ...
