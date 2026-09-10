import asyncio
import time
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from assistant.config import EnvironmentSettings, GenerationSettings, ModelProfile
from assistant.exceptions import (
    ConfigurationError,
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
                # GP owns an explicit bounded transport policy; preserve the SDK
                # default for existing baselines.
                max_retries=0 if profile.goal_progression is not None else 2,
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
        response_schema: dict[str, Any] | None = None,
    ) -> GeneratedResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        request_messages = self._messages_for_provider(messages)
        last_error: Exception | None = None
        retry = self.profile.retry
        last_finish_reason: str | None = None
        last_refusal: str | None = None
        transport_attempts: list[dict[str, Any]] = []
        if not self.profile.reasoning.enabled and self.profile.reasoning.effort is not None:
            reasoning: dict[str, Any] = {"effort": self.profile.reasoning.effort}
        else:
            reasoning = {"enabled": self.profile.reasoning.enabled}
        if self.profile.reasoning.enabled:
            if self.profile.reasoning.effort is not None:
                reasoning["effort"] = self.profile.reasoning.effort
            reasoning["exclude"] = self.profile.reasoning.exclude_from_response

        for attempt in range(1, retry.max_attempts + 1):
            started = time.perf_counter()
            attempt_response = None
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
                if response_schema is not None:
                    request_options["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "goal_progression_output",
                            "strict": True,
                            "schema": response_schema,
                        },
                    }
                if generation.temperature is not None:
                    request_options["temperature"] = generation.temperature
                if generation.top_p is not None:
                    request_options["top_p"] = generation.top_p
                if generation.presence_penalty is not None:
                    request_options["presence_penalty"] = generation.presence_penalty
                response = await self.client.chat.completions.create(**request_options)
                attempt_response = response
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
                transport_attempts.append(
                    {
                        "attempt": attempt,
                        "status": "success",
                        "latency_seconds": time.perf_counter() - started,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                )
                call_metadata = {
                    "model_id": self.profile.model_id,
                    "model_profile": self.profile.profile_name,
                    "request_id": getattr(response, "id", None),
                    "provider": getattr(response, "provider", None) or self.profile.provider,
                    "finish_reason": finish_reason,
                    "latency_seconds": time.perf_counter() - started,
                    "transport_retry_count": attempt - 1,
                    "transport_attempts": transport_attempts,
                    "input_tokens": input_tokens,
                    # OpenRouter and vLLM report total completion tokens here.
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "answer_tokens": answer_tokens,
                    "messages": request_messages,
                    "response": content,
                    "reasoning_preserved": bool(reasoning_content or reasoning_details),
                }
                self.last_call_metadata = call_metadata
                return GeneratedResponse(
                    content=content,
                    reasoning_content=reasoning_content,
                    reasoning_details=reasoning_details,
                    metadata=dict(call_metadata),
                )
            except InvalidModelResponseError as exc:
                last_error = exc
            except (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    if self.profile.goal_progression is None:
                        raise self._request_error(
                            f"rejected the request with HTTP {exc.status_code}: {exc}"
                        ) from exc
                    # Keep the server's explanation, not headers or request data.
                    body = exc.body
                    if isinstance(body, dict) and isinstance(body.get("error"), dict):
                        body = body["error"]
                    detail = body.get("message") if isinstance(body, dict) else None
                    detail = str(detail or exc.message)[:1000]
                    error = ConfigurationError(
                        f"model rejected the request with HTTP {exc.status_code}: {detail}"
                    )
                    error.call_metadata = {
                        "transport_attempts": [
                            *transport_attempts,
                            {
                                "attempt": attempt,
                                "status": "failed",
                                "http_status": exc.status_code,
                                "error_message": detail,
                                "latency_seconds": time.perf_counter() - started,
                            },
                        ]
                    }
                    raise error from exc
                last_error = exc
            transport_attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error_type": type(last_error).__name__,
                    "latency_seconds": time.perf_counter() - started,
                    "usage_status": "unknown",
                    **_failed_response_audit(attempt_response),
                }
            )
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
            "transport_attempts": transport_attempts,
            "finish_reason": last_finish_reason,
            "refusal": last_refusal,
            "error": str(last_error),
            "messages": request_messages,
        }
        if isinstance(last_error, InvalidModelResponseError):
            last_error.call_metadata = dict(self.last_call_metadata)
            raise last_error
        error = self._request_error(
            f"request failed after {retry.max_attempts} attempts: {last_error}"
        )
        error.call_metadata = dict(self.last_call_metadata)
        raise error from last_error

    async def count_visible_tokens(self, content: str) -> dict[str, Any]:
        """Tokenize the delivered text without chat templates, reasoning or EOS.

        This is a tokenizer request, not another LLM generation. Unsupported
        providers stay explicitly unknown, never substituting completion usage.
        """
        if self.profile.provider != "vllm":
            return {"visible_response_tokens": None, "visible_token_source": "unknown"}
        root = self.profile.base_url.rstrip("/").removesuffix("/v1")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    root + "/tokenize",
                    headers={"Authorization": f"Bearer {self.environment.vllm_api_key or 'EMPTY'}"},
                    json={
                        "model": self.profile.model_id,
                        "prompt": content,
                        "add_special_tokens": False,
                    },
                )
                response.raise_for_status()
                count = response.json().get("count")
                if type(count) is not int or count < 0:
                    raise ValueError("invalid tokenizer count")
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelRequestError("visible tokenizer unavailable") from exc
        return {
            "visible_response_tokens": count,
            "visible_token_source": "vllm /tokenize; add_special_tokens=false",
            "visible_token_model": self.profile.model_id,
        }

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


def _failed_response_audit(response: Any) -> dict[str, Any]:
    """Retain billed usage and visible content even for rejected/truncated output."""
    if response is None:
        return {}
    usage = getattr(response, "usage", None)
    choices = getattr(response, "choices", [])
    choice = choices[0] if choices else None
    content = getattr(getattr(choice, "message", None), "content", None)
    return {
        "request_id": getattr(response, "id", None),
        "input_tokens": _optional_nonnegative_int(getattr(usage, "prompt_tokens", None)),
        "output_tokens": _optional_nonnegative_int(getattr(usage, "completion_tokens", None)),
        "usage_status": "known" if usage is not None else "unknown",
        "finish_reason": getattr(choice, "finish_reason", None),
        "raw_output": content if isinstance(content, str) else None,
    }


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
