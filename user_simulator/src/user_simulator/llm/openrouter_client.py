import asyncio
import json
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from user_simulator.audit.logger import read_git_commit
from user_simulator.config import (
    EnvironmentSettings,
    GenerationSettings,
    ModelProfile,
    RateLimitRetrySettings,
)
from user_simulator.exceptions import (
    ModelRequestError,
    OpenRouterRequestError,
    StructuredOutputError,
)
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
        if client is not None:
            self.client = client
        else:
            api_key, headers = self._connection_settings()
            client_options: dict[str, Any] = {
                "api_key": api_key,
                "base_url": profile.base_url,
                "default_headers": headers,
            }
            if profile.retry.rate_limit is not None:
                # Keep 429 retries in this auditable client layer. Otherwise
                # the SDK adds hidden retries on top of the dedicated budget.
                client_options["max_retries"] = 0
            self.client = AsyncOpenAI(**client_options)
        self.last_call_metadata: dict[str, Any] = {}
        self.git_commit = read_git_commit()

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
        return self.environment.vllm_api_key or "EMPTY", {}

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
        structured = self.profile.structured_output
        request_schema = strict_json_schema(response_model)
        request_schema_hash = schema_hash(response_model)
        last_finish_reason: str | None = None
        last_refusal: str | None = None
        last_request_id: str | None = None
        last_provider: str | None = None
        last_latency_seconds: float | None = None
        last_invalid_response: Any | None = None
        last_invalid_content: str | None = None
        transport_retry_count = 0
        structured_retry_count = 0
        rate_limit_attempt_count = 0
        rate_limit_retry_count = 0
        rate_limit_wait_seconds = 0.0
        last_rate_limit_type: str | None = None
        total_attempt_count = 0
        ordinary_attempt_count = 0
        structured_correction = ""
        request_messages = messages
        reasoning_settings = generation.reasoning or self.profile.reasoning
        reasoning: dict[str, Any] = {"enabled": reasoning_settings.enabled}
        if reasoning_settings.enabled:
            if reasoning_settings.effort is not None:
                reasoning["effort"] = reasoning_settings.effort
            reasoning["exclude"] = reasoning_settings.exclude_from_response
        while True:
            total_attempt_count += 1
            started = time.perf_counter()
            request_messages = _with_structured_correction(
                messages,
                structured_correction,
                last_invalid_content,
            )
            if structured.transport == "json_object":
                request_messages = _with_json_object_contract(
                    request_messages,
                    schema_name=schema_name,
                    request_schema=request_schema,
                )
            retry_kind = "structured"
            content: Any = None
            try:
                extra_body: dict[str, Any] = {}
                if self.profile.provider == "openrouter":
                    provider_routing = self.profile.routing.model_dump(exclude_none=True)
                    extra_body.update(
                        {
                            "provider": {
                                **provider_routing,
                                "require_parameters": structured.require_parameters,
                            },
                            "reasoning": reasoning,
                        }
                    )
                elif reasoning_settings.local_chat_template is not None:
                    extra_body["chat_template_kwargs"] = (
                        reasoning_settings.local_chat_template.model_dump(exclude_none=True)
                    )
                if structured.response_healing and self.profile.provider == "openrouter":
                    extra_body["plugins"] = [{"id": "response-healing"}]
                for key in ("top_k", "min_p", "repetition_penalty"):
                    value = getattr(generation, key)
                    if value is not None:
                        extra_body[key] = value
                response_format: dict[str, Any] = {"type": structured.transport}
                if structured.transport == "json_schema":
                    response_format["json_schema"] = {
                        "name": schema_name,
                        "strict": structured.strict,
                        "schema": request_schema,
                    }
                request_options: dict[str, Any] = {
                    "model": self.profile.model_id,
                    "messages": request_messages,
                    # OpenRouter currently advertises this parameter as
                    # ``max_tokens`` for the configured profiles. Sending
                    # ``max_completion_tokens`` together with
                    # provider.require_parameters=true filters out every
                    # otherwise-compatible endpoint.
                    "max_tokens": generation.max_completion_tokens,
                    "stream": False,
                    "response_format": response_format,
                    "extra_body": extra_body,
                }
                # Some reasoning models (including GPT-5.6 Luna on OpenRouter)
                # do not advertise temperature support. ``None`` means omit the
                # parameter so require_parameters can still find an endpoint.
                if generation.temperature is not None:
                    request_options["temperature"] = generation.temperature
                if generation.top_p is not None:
                    request_options["top_p"] = generation.top_p
                if generation.presence_penalty is not None:
                    request_options["presence_penalty"] = generation.presence_penalty
                response = await self.client.chat.completions.create(**request_options)
                last_request_id = getattr(response, "id", None)
                last_provider = getattr(response, "provider", None) or self.profile.provider
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
                input_tokens = _optional_nonnegative_int(
                    getattr(usage, "prompt_tokens", None)
                )
                output_tokens = _optional_nonnegative_int(
                    getattr(usage, "completion_tokens", None)
                )
                completion_token_details = getattr(
                    usage, "completion_tokens_details", None
                )
                thinking_tokens = _optional_nonnegative_int(
                    getattr(completion_token_details, "reasoning_tokens", None)
                )
                answer_tokens = _answer_token_count(output_tokens, thinking_tokens)
                last_latency_seconds = time.perf_counter() - started
                self.last_call_metadata = {
                    **(prompt_metadata or {}),
                    "model_id": self.profile.model_id,
                    "model_profile": self.profile.profile_name,
                    "git_commit": self.git_commit,
                    "schema_name": schema_name,
                    "schema_hash": request_schema_hash,
                    "structured_output_type": structured.type,
                    "structured_output_transport": structured.transport,
                    "provider_schema_enforced": structured.transport == "json_schema",
                    "local_schema_validation": True,
                    "strict": structured.strict,
                    "require_parameters": structured.require_parameters,
                    "response_healing": structured.response_healing,
                    "reasoning": reasoning,
                    "request_id": last_request_id,
                    "provider": last_provider,
                    "finish_reason": finish_reason,
                    "latency_seconds": last_latency_seconds,
                    "request_attempt_count": total_attempt_count,
                    "transport_retry_count": transport_retry_count,
                    "structured_retry_count": structured_retry_count,
                    "rate_limit_retry_count": rate_limit_retry_count,
                    "rate_limit_wait_seconds": rate_limit_wait_seconds,
                    "last_rate_limit_type": last_rate_limit_type,
                    "structured_validation_status": "valid",
                    "input_tokens": input_tokens,
                    # OpenRouter includes reasoning/thinking tokens in completion_tokens.
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "answer_tokens": answer_tokens,
                    "messages": request_messages,
                    "raw_response": result.model_dump(mode="json"),
                }
                if last_invalid_response is not None:
                    self.last_call_metadata["invalid_raw_response"] = last_invalid_response
                return result
            except ValidationError as exc:
                last_invalid_response = _parse_invalid_response(content)
                last_invalid_content = _structured_retry_excerpt(content)
                detail = _structured_error_detail(exc)
                last_error = StructuredOutputError(detail)
                structured_correction = _structured_correction(detail)
            except (StructuredOutputError, ValueError, KeyError, IndexError) as exc:
                if content is not None:
                    last_invalid_response = _parse_invalid_response(content)
                    last_invalid_content = _structured_retry_excerpt(content)
                else:
                    last_invalid_content = None
                detail = str(exc)
                last_error = StructuredOutputError(detail)
                structured_correction = _structured_correction(detail)
            except RateLimitError as exc:
                last_error = exc
                retry_kind = "rate_limit"
                last_provider = _rate_limit_provider(exc) or last_provider
            except (APIConnectionError, APITimeoutError, TimeoutError) as exc:
                last_error = exc
                retry_kind = "transport"
            except APIStatusError as exc:
                if exc.status_code == 429:
                    last_error = exc
                    retry_kind = "rate_limit"
                    last_provider = _rate_limit_provider(exc) or last_provider
                elif exc.status_code < 500:
                    raise self._request_error(
                        f"rejected the request with HTTP {exc.status_code}: {exc}"
                    ) from exc
                else:
                    last_error = exc
                    retry_kind = "transport"
            last_latency_seconds = time.perf_counter() - started
            if retry_kind == "rate_limit" and retry.rate_limit is not None:
                rate_limit_attempt_count += 1
                last_rate_limit_type = _rate_limit_type(last_error)
                if rate_limit_attempt_count >= retry.rate_limit.max_attempts:
                    break
                rate_limit_retry_count += 1
                transport_retry_count += 1
                delay = _rate_limit_retry_delay(
                    retry.rate_limit,
                    last_rate_limit_type,
                    rate_limit_retry_count,
                    last_error,
                )
                rate_limit_wait_seconds += delay
                await asyncio.sleep(delay)
                continue

            ordinary_attempt_count += 1
            if ordinary_attempt_count >= retry.max_attempts:
                break
            if retry_kind in {"transport", "rate_limit"}:
                transport_retry_count += 1
            else:
                structured_retry_count += 1
            delay = min(
                retry.initial_backoff_seconds * (2 ** (ordinary_attempt_count - 1)),
                retry.maximum_backoff_seconds,
            )
            await asyncio.sleep(delay)
        structured_validation_status = (
            "invalid" if isinstance(last_error, StructuredOutputError) else "not_run"
        )
        self.last_call_metadata = {
            **(prompt_metadata or {}),
            "model_id": self.profile.model_id,
            "model_profile": self.profile.profile_name,
            "git_commit": self.git_commit,
            "schema_name": schema_name,
            "schema_hash": request_schema_hash,
            "structured_output_type": structured.type,
            "structured_output_transport": structured.transport,
            "provider_schema_enforced": structured.transport == "json_schema",
            "local_schema_validation": True,
            "strict": structured.strict,
            "require_parameters": structured.require_parameters,
            "response_healing": structured.response_healing,
            "reasoning": reasoning,
            "request_id": last_request_id,
            "provider": last_provider,
            "latency_seconds": last_latency_seconds,
            "request_attempt_count": total_attempt_count,
            "transport_retry_count": transport_retry_count,
            "structured_retry_count": structured_retry_count,
            "rate_limit_retry_count": rate_limit_retry_count,
            "rate_limit_wait_seconds": rate_limit_wait_seconds,
            "last_rate_limit_type": last_rate_limit_type,
            "structured_validation_status": structured_validation_status,
            "finish_reason": last_finish_reason,
            "refusal": last_refusal,
            "validation_error": str(last_error),
            "messages": request_messages,
        }
        if last_invalid_response is not None:
            self.last_call_metadata["invalid_raw_response"] = last_invalid_response
        if isinstance(last_error, StructuredOutputError):
            raise last_error
        raise self._request_error(
            f"request failed after {total_attempt_count} attempts: {last_error}"
        ) from last_error

    def _request_error(self, detail: str) -> ModelRequestError:
        if self.profile.provider == "openrouter":
            return OpenRouterRequestError(f"OpenRouter {detail}")
        return ModelRequestError(f"Local OpenAI-compatible server {detail}")


def _rate_limit_metadata(exc: Exception | None) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}
    error = body.get("error", body)
    if not isinstance(error, dict):
        return {}
    metadata = error.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _rate_limit_type(exc: Exception | None) -> str:
    metadata = _rate_limit_metadata(exc)
    provider_code = str(metadata.get("provider_error_code", "")).lower()
    searchable = f"{provider_code} {getattr(exc, 'body', '')} {exc}".lower()
    if "rpm_rate_limit_exceeded" in searchable or "rpm" in provider_code:
        return "rpm"
    if "tpm_rate_limit_exceeded" in searchable or "tpm" in provider_code:
        return "tpm"
    return "generic"


def _rate_limit_provider(exc: Exception | None) -> str | None:
    provider = _rate_limit_metadata(exc).get("provider_name")
    return str(provider) if provider else None


def _retry_after_seconds(exc: Exception | None) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw_value = headers.get("retry-after")
    if raw_value is None:
        return None
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw_value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _rate_limit_retry_delay(
    settings: RateLimitRetrySettings,
    limit_type: str,
    retry_number: int,
    exc: Exception | None,
) -> float:
    initial = {
        "rpm": settings.rpm_initial_backoff_seconds,
        "tpm": settings.tpm_initial_backoff_seconds,
    }.get(limit_type, settings.generic_initial_backoff_seconds)
    exponential_delay = min(
        initial * (2 ** (retry_number - 1)),
        settings.maximum_backoff_seconds,
    )
    delay_floor = exponential_delay
    if settings.honor_retry_after:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay_floor = max(delay_floor, retry_after)
    jitter = random.uniform(0.0, delay_floor * settings.jitter_ratio)
    return delay_floor + jitter


def _with_structured_correction(
    messages: list[dict[str, str]],
    correction: str,
    invalid_content: str | None = None,
) -> list[dict[str, str]]:
    copied = [dict(message) for message in messages]
    if not correction:
        return copied
    if invalid_content:
        copied.append({"role": "assistant", "content": invalid_content})
    copied.append({"role": "user", "content": correction})
    return copied


def _with_json_object_contract(
    messages: list[dict[str, str]],
    *,
    schema_name: str,
    request_schema: dict[str, Any],
) -> list[dict[str, str]]:
    """Expose the local schema to JSON Object endpoints without mutating prompts."""

    schema_json = json.dumps(
        request_schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    instruction = (
        "JSON OBJECT CONTRACT\n"
        "Return only one valid JSON object matching the following JSON Schema exactly. "
        "Include every required field, obey enum and null constraints, and add no extra "
        f"fields. Contract name: {schema_name}. JSON Schema: {schema_json}"
    )
    copied = [dict(message) for message in messages]
    for index, message in enumerate(copied):
        if message.get("role") != "system":
            continue
        copied[index]["content"] = f"{message.get('content', '').rstrip()}\n\n{instruction}"
        return copied
    return [{"role": "system", "content": instruction}, *copied]


def _structured_error_detail(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False, include_input=False)
    return json.dumps(errors, ensure_ascii=False, separators=(",", ":"))[:4000]


def _structured_correction(detail: str) -> str:
    return (
        "STRUCTURED OUTPUT CORRECTION\n"
        f"The previous response failed local schema validation: {detail}\n"
        "Return only a complete replacement JSON object under the same enforced schema. "
        "Use exactly the schema field names, include every required top-level and nested "
        "field, add no extra fields, and obey all enum and null/value combinations. Do not "
        "discuss the error or repeat the invalid field combination."
    )


def _structured_retry_excerpt(content: Any) -> str | None:
    if not isinstance(content, str) or not content.strip():
        return None
    return content[:8000]


def _parse_invalid_response(content: Any) -> Any:
    if not isinstance(content, str):
        return "<non-text structured response omitted>"
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content[:4000]


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
