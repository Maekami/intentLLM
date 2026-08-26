import asyncio
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from assistant.config import EnvironmentSettings, GenerationSettings, ModelProfile
from assistant.exceptions import (
    InvalidModelResponseError,
    ModelRequestError,
    OpenRouterRequestError,
)
from assistant.llm.base import GeneratedResponse


class OpenAICompatibleChatClient:
    def __init__(
        self,
        profile: ModelProfile,
        environment: EnvironmentSettings | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.profile = profile
        self.environment = environment or EnvironmentSettings()
        if client is not None:
            self.client = client
        else:
            api_key, headers = self._connection_settings()
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=profile.base_url,
                default_headers=headers,
            )
        self.last_call_metadata: dict[str, Any] = {}

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if self.profile.provider == "openrouter":
            if not self.environment.openrouter_api_key:
                raise OpenRouterRequestError("OPENROUTER_API_KEY is not set")
            headers = {
                key: value
                for key, value in {
                    "HTTP-Referer": self.environment.openrouter_http_referer,
                    "X-Title": self.environment.openrouter_app_title,
                }.items()
                if value
            }
            return self.environment.openrouter_api_key, headers
        # The OpenAI SDK requires a non-empty key even when an unauthenticated
        # local vLLM server ignores the Authorization header.
        return self.environment.vllm_api_key or "EMPTY", {}

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        generation: GenerationSettings,
    ) -> GeneratedResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        request_messages = self._messages_for_provider(messages)
        last_error: Exception | None = None
        retry = self.profile.retry
        last_finish_reason: str | None = None
        last_refusal: str | None = None
        if (
            not self.profile.reasoning.enabled
            and self.profile.reasoning.effort is not None
        ):
            reasoning: dict[str, Any] = {"effort": self.profile.reasoning.effort}
        else:
            reasoning = {"enabled": self.profile.reasoning.enabled}
        if self.profile.reasoning.enabled:
            if self.profile.reasoning.effort is not None:
                reasoning["effort"] = self.profile.reasoning.effort
            reasoning["exclude"] = self.profile.reasoning.exclude_from_response

        for attempt in range(1, retry.max_attempts + 1):
            started = time.perf_counter()
            try:
                extra_body: dict[str, Any] = {}
                if self.profile.provider == "openrouter":
                    extra_body.update(
                        {
                            "provider": self.profile.routing,
                            "reasoning": reasoning,
                        }
                    )
                elif self.profile.reasoning.local_chat_template is not None:
                    extra_body["chat_template_kwargs"] = (
                        self.profile.reasoning.local_chat_template.model_dump()
                    )
                for key in ("top_k", "min_p", "repetition_penalty"):
                    value = getattr(generation, key)
                    if value is not None:
                        extra_body[key] = value

                request_options: dict[str, Any] = {
                    "model": self.profile.model_id,
                    "messages": request_messages,
                    "max_tokens": generation.max_completion_tokens,
                    "stream": False,
                    "extra_body": extra_body,
                }
                if generation.temperature is not None:
                    request_options["temperature"] = generation.temperature
                if generation.top_p is not None:
                    request_options["top_p"] = generation.top_p
                if generation.presence_penalty is not None:
                    request_options["presence_penalty"] = generation.presence_penalty
                response = await self.client.chat.completions.create(**request_options)
                if not response.choices:
                    raise InvalidModelResponseError("model response contained no choices")
                choice = response.choices[0]
                message = choice.message
                refusal = getattr(message, "refusal", None)
                finish_reason = getattr(choice, "finish_reason", None)
                last_refusal = str(refusal) if refusal else None
                last_finish_reason = finish_reason
                if refusal:
                    raise InvalidModelResponseError(f"model refused the request: {refusal}")
                if finish_reason == "length":
                    raise InvalidModelResponseError(
                        "model output was truncated at the configured token limit"
                    )
                content = message.content
                if not isinstance(content, str) or not content.strip():
                    raise InvalidModelResponseError("model returned an empty text response")
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content is None:
                    reasoning_content = getattr(message, "reasoning", None)
                if not isinstance(reasoning_content, str) or not reasoning_content.strip():
                    reasoning_content = None
                reasoning_details = _json_value(getattr(message, "reasoning_details", None))
                if not isinstance(reasoning_details, list):
                    reasoning_details = None
                usage = getattr(response, "usage", None)
                input_tokens = _optional_nonnegative_int(getattr(usage, "prompt_tokens", None))
                output_tokens = _optional_nonnegative_int(getattr(usage, "completion_tokens", None))
                completion_token_details = getattr(usage, "completion_tokens_details", None)
                thinking_tokens = _optional_nonnegative_int(
                    getattr(completion_token_details, "reasoning_tokens", None)
                )
                answer_tokens = _answer_token_count(output_tokens, thinking_tokens)
                self.last_call_metadata = {
                    "model_id": self.profile.model_id,
                    "model_profile": self.profile.profile_name,
                    "request_id": getattr(response, "id", None),
                    "provider": getattr(response, "provider", None) or self.profile.provider,
                    "finish_reason": finish_reason,
                    "latency_seconds": time.perf_counter() - started,
                    "transport_retry_count": attempt - 1,
                    "input_tokens": input_tokens,
                    # OpenRouter and vLLM report total completion tokens here.
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "answer_tokens": answer_tokens,
                    "messages": request_messages,
                    "response": content,
                    "reasoning_preserved": bool(reasoning_content or reasoning_details),
                }
                return GeneratedResponse(
                    content=content,
                    reasoning_content=reasoning_content,
                    reasoning_details=reasoning_details,
                )
            except InvalidModelResponseError as exc:
                last_error = exc
            except (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise self._request_error(
                        f"rejected the request with HTTP {exc.status_code}: {exc}"
                    ) from exc
                last_error = exc
            if attempt < retry.max_attempts:
                delay = min(
                    retry.initial_backoff_seconds * (2 ** (attempt - 1)),
                    retry.maximum_backoff_seconds,
                )
                await asyncio.sleep(delay)

        self.last_call_metadata = {
            "model_id": self.profile.model_id,
            "model_profile": self.profile.profile_name,
            "transport_retry_count": retry.max_attempts - 1,
            "finish_reason": last_finish_reason,
            "refusal": last_refusal,
            "error": str(last_error),
            "messages": request_messages,
        }
        if isinstance(last_error, InvalidModelResponseError):
            raise last_error
        raise self._request_error(
            f"request failed after {retry.max_attempts} attempts: {last_error}"
        ) from last_error

    def _request_error(self, detail: str) -> ModelRequestError:
        if self.profile.provider == "openrouter":
            return OpenRouterRequestError(f"OpenRouter {detail}")
        return ModelRequestError(f"Local OpenAI-compatible server {detail}")

    def _messages_for_provider(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.profile.provider == "openrouter":
            return messages
        normalized: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            legacy_reasoning = item.pop("reasoning_content", None)
            if legacy_reasoning is not None and item.get("reasoning") is None:
                # Current vLLM names the request/response field `reasoning`.
                # Normalize explicitly instead of relying on its deprecated
                # reasoning_content compatibility path.
                item["reasoning"] = legacy_reasoning
            normalized.append(item)
        return normalized


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    return None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _answer_token_count(
    output_tokens: int | None,
    thinking_tokens: int | None,
) -> int | None:
    if output_tokens is None or thinking_tokens is None:
        return None
    answer_tokens = output_tokens - thinking_tokens
    return answer_tokens if answer_tokens >= 0 else None


# Preserve the public name used by existing imports while the implementation
# now supports both OpenRouter and local OpenAI-compatible servers.
OpenRouterChatClient = OpenAICompatibleChatClient
