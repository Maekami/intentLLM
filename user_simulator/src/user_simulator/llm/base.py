from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from user_simulator.config import GenerationSettings

T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient(Protocol):
    async def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[T],
        schema_name: str,
        generation: GenerationSettings,
        prompt_metadata: dict[str, Any] | None = None,
    ) -> T: ...
