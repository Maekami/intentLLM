import asyncio
import json
import time
from typing import Any, TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from user_simulator.config import EnvironmentSettings, GenerationSettings, ModelProfile
from user_simulator.exceptions import OpenRouterRequestError, StructuredOutputError

T = TypeVar("T", bound=BaseModel)


class OpenRouterStructuredClient:
    def __init__(
        self,
        profile: ModelProfile,
        environment: EnvironmentSettings | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.profile = profile
        self.environment = environment or EnvironmentSettings()
        if not self.environment.openrouter_api_key and client is None:
            raise OpenRouterRequestError("OPENROUTER_API_KEY is not set")
        headers = {
            key: value
            for key, value in {
                "HTTP-Referer": self.environment.openrouter_http_referer,
                "X-Title": self.environment.openrouter_app_title,
            }.items()
            if value
        }
        self.client = client or AsyncOpenAI(
            api_key=self.environment.openrouter_api_key,
            base_url=profile.base_url,
            default_headers=headers,
        )
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
        last_error: Exception | None = None
        retry = self.profile.retry
        for attempt in range(1, retry.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = await self.client.chat.completions.create(
                    model=self.profile.model_id,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=generation.temperature,
                    max_completion_tokens=generation.max_completion_tokens,
                    stream=False,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": response_model.model_json_schema(),
                        },
                    },
                    extra_body={
                        "provider": self.profile.routing,
                        "reasoning": {
                            "effort": self.profile.reasoning.get("effort", "high"),
                            "exclude": self.profile.reasoning.get("exclude_from_response", True),
                        },
                    },
                )
                content = response.choices[0].message.content
                if not content:
                    raise StructuredOutputError("model returned empty structured output")
                result = response_model.model_validate(json.loads(content))
                usage = getattr(response, "usage", None)
                self.last_call_metadata = {
                    **(prompt_metadata or {}),
                    "component_schema": schema_name,
                    "model_id": self.profile.model_id,
                    "request_id": getattr(response, "id", None),
                    "latency_seconds": time.perf_counter() - started,
                    "retry_count": attempt - 1,
                    "structured_validation_status": "valid",
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "provider_metadata": getattr(response, "provider", None),
                    "messages": messages,
                    "raw_response": result.model_dump(mode="json"),
                }
                return result
            except (
                StructuredOutputError,
                ValidationError,
                ValueError,
                KeyError,
                IndexError,
            ) as exc:
                last_error = StructuredOutputError(str(exc))
            except (OpenAIError, TimeoutError) as exc:
                last_error = exc
            if attempt < retry.max_attempts:
                delay = min(
                    retry.initial_backoff_seconds * (2 ** (attempt - 1)),
                    retry.maximum_backoff_seconds,
                )
                await asyncio.sleep(delay)
        self.last_call_metadata = {
            **(prompt_metadata or {}),
            "component_schema": schema_name,
            "model_id": self.profile.model_id,
            "retry_count": retry.max_attempts - 1,
            "structured_validation_status": "invalid",
            "messages": messages,
        }
        if isinstance(last_error, StructuredOutputError):
            raise last_error
        raise OpenRouterRequestError(
            f"OpenRouter request failed after {retry.max_attempts} attempts: {last_error}"
        ) from last_error
