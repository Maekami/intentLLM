import asyncio
import time
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from user_simulator.config import EnvironmentSettings, GenerationSettings, ModelProfile
from user_simulator.exceptions import OpenRouterRequestError, StructuredOutputError
from user_simulator.llm.schema_utils import schema_hash, strict_json_schema

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
        schema_version: int = 2,
        generation: GenerationSettings,
        prompt_metadata: dict[str, Any] | None = None,
    ) -> T:
        last_error: Exception | None = None
        retry = self.profile.retry
        structured = self.profile.structured_output
        request_schema = strict_json_schema(response_model)
        request_schema_hash = schema_hash(response_model)
        last_finish_reason: str | None = None
        last_refusal: str | None = None
        for attempt in range(1, retry.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = await self.client.chat.completions.create(
                    model=self.profile.model_id,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=generation.temperature,
                    # OpenRouter currently advertises this parameter as
                    # ``max_tokens`` for DeepSeek V4 Pro. Sending
                    # ``max_completion_tokens`` together with
                    # provider.require_parameters=true filters out every
                    # otherwise-compatible endpoint.
                    max_tokens=generation.max_completion_tokens,
                    stream=False,
                    response_format={
                        "type": structured.type,
                        "json_schema": {
                            "name": schema_name,
                            "strict": structured.strict,
                            "schema": request_schema,
                        },
                    },
                    extra_body={
                        "provider": {
                            **self.profile.routing,
                            "require_parameters": structured.require_parameters,
                        },
                        "reasoning": {
                            "effort": self.profile.reasoning.get("effort", "high"),
                            "exclude": self.profile.reasoning.get("exclude_from_response", True),
                        },
                    },
                )
                if not response.choices:
                    raise StructuredOutputError("model response contained no choices")
                choice = response.choices[0]
                message = choice.message
                refusal = getattr(message, "refusal", None)
                finish_reason = getattr(choice, "finish_reason", None)
                last_refusal = str(refusal) if refusal else None
                last_finish_reason = finish_reason
                if refusal:
                    raise StructuredOutputError(f"model refused the request: {refusal}")
                if finish_reason == "length":
                    raise StructuredOutputError("model output was truncated at token limit")
                content = message.content
                if not isinstance(content, str) or not content.strip():
                    raise StructuredOutputError("model returned empty structured output")
                result = response_model.model_validate_json(content)
                usage = getattr(response, "usage", None)
                self.last_call_metadata = {
                    **(prompt_metadata or {}),
                    "model_id": self.profile.model_id,
                    "model_profile": self.profile.profile_name,
                    "schema_name": schema_name,
                    "schema_version": schema_version,
                    "schema_hash": request_schema_hash,
                    "structured_output_type": structured.type,
                    "strict": structured.strict,
                    "require_parameters": structured.require_parameters,
                    "request_id": getattr(response, "id", None),
                    "provider": getattr(response, "provider", None),
                    "finish_reason": finish_reason,
                    "latency_seconds": time.perf_counter() - started,
                    "transport_retry_count": attempt - 1,
                    "structured_validation_status": "valid",
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
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
            except (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise OpenRouterRequestError(
                        f"OpenRouter rejected the request with HTTP {exc.status_code}: {exc}"
                    ) from exc
                last_error = exc
            if attempt < retry.max_attempts:
                delay = min(
                    retry.initial_backoff_seconds * (2 ** (attempt - 1)),
                    retry.maximum_backoff_seconds,
                )
                await asyncio.sleep(delay)
        self.last_call_metadata = {
            **(prompt_metadata or {}),
            "model_id": self.profile.model_id,
            "model_profile": self.profile.profile_name,
            "schema_name": schema_name,
            "schema_version": schema_version,
            "schema_hash": request_schema_hash,
            "structured_output_type": structured.type,
            "strict": structured.strict,
            "require_parameters": structured.require_parameters,
            "transport_retry_count": retry.max_attempts - 1,
            "structured_validation_status": "invalid",
            "finish_reason": last_finish_reason,
            "refusal": last_refusal,
            "validation_error": str(last_error),
            "messages": messages,
        }
        if isinstance(last_error, StructuredOutputError):
            raise last_error
        raise OpenRouterRequestError(
            f"OpenRouter request failed after {retry.max_attempts} attempts: {last_error}"
        ) from last_error
