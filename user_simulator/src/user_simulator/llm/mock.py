import json
from collections import deque
from typing import Any, TypeVar

from pydantic import BaseModel

from user_simulator.audit.logger import read_git_commit
from user_simulator.config import GenerationSettings
from user_simulator.exceptions import StructuredOutputError
from user_simulator.llm.schema_utils import schema_hash

T = TypeVar("T", bound=BaseModel)


class MockStructuredLLMClient:
    """Deterministic queued client for unit and integration tests."""

    def __init__(self, responses: list[BaseModel | dict[str, Any]]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []
        self.last_call_metadata: dict[str, Any] = {}

    async def generate_structured(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[T],
        schema_name: str,
        generation: GenerationSettings,
        prompt_metadata: dict[str, Any] | None = None,
    ) -> T:
        self.calls.append(
            {
                "messages": messages,
                "response_model": response_model.__name__,
                "schema_name": schema_name,
                "generation": generation.model_dump(),
                "prompt_metadata": prompt_metadata or {},
            }
        )
        self.last_call_metadata = {
            **(prompt_metadata or {}),
            "schema_name": schema_name,
            "schema_hash": schema_hash(response_model),
            "model_id": "mock",
            "model_profile": "mock",
            "git_commit": read_git_commit(),
            "transport_retry_count": 0,
            "structured_validation_status": "valid",
            "messages": messages,
        }
        if not self.responses:
            raise StructuredOutputError("mock response queue is empty")
        value = self.responses.popleft()
        if isinstance(value, BaseModel):
            value = value.model_dump()
        result = response_model.model_validate_json(json.dumps(value))
        self.last_call_metadata["raw_response"] = result.model_dump(mode="json")
        return result
