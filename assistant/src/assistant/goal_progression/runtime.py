"""One error/retry runtime for every GP producer; no nested transport retry loop."""

import asyncio
import copy
import json
import math
import time
import weakref

from pydantic import ValidationError

from assistant.config import GPRoleBudget
from assistant.exceptions import ConfigurationError, InvalidModelResponseError, ModelRequestError

from .contracts import validate_wire_required
from .errors import ContractError, RecoveryExhausted
from .events import digest
from .repair_context import repair_excerpt

_LIMITERS = weakref.WeakKeyDictionary()
_USAGE_KEYS = {
    "model_id",
    "model_profile",
    "request_id",
    "provider",
    "finish_reason",
    "latency_seconds",
    "transport_retry_count",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "answer_tokens",
    "transport_attempts",
}


def provider_limiter(profile):
    registry = _LIMITERS.setdefault(asyncio.get_running_loop(), {})
    key = (profile.provider, profile.base_url.rstrip("/"), profile.model_id)
    limit = profile.goal_progression.max_in_flight_requests
    if key not in registry:
        registry[key] = (limit, asyncio.Semaphore(limit))
    if registry[key][0] != limit:
        raise ConfigurationError("conflicting GP in-flight limits for the same provider/model")
    return registry[key][1]


def public_usage(metadata):
    result = {
        key: copy.deepcopy(value)
        for key, value in (metadata or {}).items()
        if key in _USAGE_KEYS and key != "transport_attempts"
    }
    attempts = (metadata or {}).get("transport_attempts")
    if isinstance(attempts, list):
        allowed = (_USAGE_KEYS - {"transport_attempts"}) | {
            "attempt",
            "status",
            "error_type",
            "error_message",
            "http_status",
            "usage_status",
            "raw_output",
        }
        result["transport_attempts"] = [
            {key: copy.deepcopy(value) for key, value in attempt.items() if key in allowed}
            for attempt in attempts
            if isinstance(attempt, dict)
        ]
        for name in ("input_tokens", "output_tokens"):
            counts = [a.get(name) for a in result["transport_attempts"]]
            if (
                type(result.get(name)) is not int
                and counts
                and all(type(n) is int and n >= 0 for n in counts)
            ):
                result[name] = sum(counts)
    return result


def classify_exception(exc, role):
    if isinstance(exc, ContractError):
        return exc
    metadata = getattr(exc, "call_metadata", {}) or {}
    cause = exc
    names = set()
    while cause is not None and type(cause).__name__ not in names:
        names.add(type(cause).__name__)
        cause = cause.__cause__
    attempts = metadata.get("transport_attempts", [])
    names.update(a.get("error_type") for a in attempts if isinstance(a, dict))
    if metadata.get("finish_reason") == "length":
        return ContractError("output.truncated", role, "Provider reported output truncation")
    if metadata.get("refusal"):
        raise InvalidModelResponseError("GP role explicitly refused; not a format retry") from exc
    if "RateLimitError" in names:
        return ContractError("call.rate_limit", role, "Provider rate limited this attempt")
    if names & {"TimeoutError", "APITimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return ContractError("call.timeout", role, "Model request timed out")
    if isinstance(exc, ConfigurationError):
        cause = exc.__cause__
        body = getattr(cause, "body", None)
        if isinstance(body, dict):
            detail = body.get("error", body)
            if isinstance(detail, dict) and detail.get("code") in {
                "context_length_exceeded",
                "max_context_length_exceeded",
            }:
                return ContractError(
                    "context.capacity",
                    role,
                    "Provider context capacity exceeded",
                    recoverable=False,
                )
        raise exc
    if isinstance(exc, InvalidModelResponseError):
        return ContractError("output.empty", role, "Provider returned no usable content")
    if isinstance(exc, ModelRequestError):
        return ContractError("call.transient", role, "Recoverable model request failure")
    raise exc


class ContractRuntime:
    def __init__(self, client, profile, prompts, recorder, ledger, audit):
        self.client, self.profile, self.prompts = client, profile, prompts
        self.settings = profile.goal_progression
        self.recorder, self.ledger, self.audit = recorder, ledger, audit
        self.semaphore = provider_limiter(profile)
        self.expanded = set()
        self.output_limits = {}
        self.started = time.perf_counter()

    def repair(self, error):
        if error.artifact_id is None:
            error.artifact_id = f"{self.recorder.turn_id}:{error.owner}"
        if error.input_version is None:
            error.input_version = self.recorder.turn_id
        try:
            counts = self.ledger.consume([error])
        except RecoveryExhausted:
            self.recorder.emit(
                "assistant_gp_recovery_exhausted",
                {
                    **error.payload(),
                    "retry_counts": self.ledger.snapshot(),
                },
            )
            raise
        self.recorder.emit(
            "assistant_gp_retry_scheduled",
            {
                **error.payload(),
                "retry_counts": counts,
            },
        )
        if error.code.startswith("call."):
            instruction = (
                "The request failed before a usable response. Retry the same task and schema; "
                "this is not evidence that user state or an upstream plan is wrong."
            )
        elif error.code == "output.truncated":
            instruction = (
                "The output was cut off. Regenerate a complete, compact JSON object for "
                "your owned role. Do not continue the partial string, loop over alternatives, "
                "or write search/deliberation. Keep required fields and substantive "
                "deliverables; remove repeated or meta text."
            )
        else:
            instruction = (
                "Correct the reported fields using their stated constraints. Update linked "
                "references consistently, preserve unaffected work, and return only your owned JSON. "
                "Do not repeat the rejected output unchanged."
            )
        return {"errors": [error.payload()], "instruction": instruction}

    def generation(self, role):
        key = "assistant" if role == "generator" else role
        result = self.profile.generation[key].model_copy(deep=True)
        if role in self.output_limits:
            result.max_completion_tokens = self.output_limits[role]
        return result

    async def preflight(self, role, messages, generation):
        context = self.settings.context
        budget = context.role_budgets.get(role, GPRoleBudget())
        # Deliberately conservative and labelled, not presented as actual Qwen token usage.
        estimate = sum(len(m["content"].encode("utf-8")) for m in messages)
        source = "utf8_upper_bound"
        counter = getattr(self.client, "count_visible_tokens", None)
        if context.estimator == "provider_text_with_overhead" and counter is not None:
            try:
                async with self.semaphore:
                    async with asyncio.timeout(self.settings.recovery.call_timeout_seconds):
                        counted = await counter("\n".join(m["content"] for m in messages))
                count = counted.get("visible_response_tokens")
                if type(count) is int and count >= 0:
                    estimate, source = count, "provider_text_with_chat_overhead"
            except (ModelRequestError, TimeoutError, ValueError):
                self.recorder.emit(
                    "assistant_gp_context_tokenizer_unavailable",
                    {
                        "role": role,
                        "fallback_estimator": "utf8_upper_bound",
                    },
                )
        estimate += context.chat_overhead_tokens
        soft = budget.expanded_input_tokens if role in self.expanded else budget.max_input_tokens
        hard = context.hard_context_tokens
        hard_exceeded = hard is not None and estimate + generation.max_completion_tokens > hard
        role_exceeded = soft is not None and estimate > soft
        self.recorder.emit(
            "assistant_gp_context_preflight",
            {
                "role": role,
                "input_token_estimate": estimate,
                "input_estimator": source,
                "output_token_reserve": generation.max_completion_tokens,
                "role_input_limit": soft,
                "expanded_input_limit": budget.expanded_input_tokens,
                "expanded": role in self.expanded,
                "hard_context_limit": hard,
                "status": "hard_limit" if hard_exceeded else "role_limit" if role_exceeded else "ok",
            },
        )
        if hard_exceeded:
            raise ContractError(
                "context.capacity",
                role,
                f"Required input estimate {estimate} plus output reserve "
                f"{generation.max_completion_tokens} exceeds hard capacity {hard}",
                recoverable=False,
            )
        if role_exceeded:
            can_expand = (
                role not in self.expanded
                and budget.expanded_input_tokens is not None
                and estimate <= budget.expanded_input_tokens
            )
            raise ContractError(
                "context.capacity",
                role,
                f"Required input estimate {estimate} exceeds role budget {soft}; "
                f"configured expansion limit is {budget.expanded_input_tokens}",
                recoverable=can_expand,
            )
        return estimate, source

    async def call(
        self,
        role,
        model,
        context,
        schema,
        *,
        validate=None,
        correction=None,
        mode="normal",
        context_factory=None,
    ):
        context = copy.deepcopy(context)
        raw = None
        while True:
            if context_factory is not None:
                context, schema = context_factory()
                context = copy.deepcopy(context)
            prompt = self.prompts[role]
            instruction = prompt.system_for(mode)
            instruction += "\n\n## Output schema\n" + json.dumps(schema, ensure_ascii=False)
            payload = {"role": role, "mode": mode, "context": context}
            if correction is not None:
                instruction += (
                    "\n\n## Repair attempt\nThe repair block identifies a rejected contract "
                    "field. Correct that defect against the supplied input facts and schema; "
                    "the rejected excerpt is NOT an example to copy. Preserve unaffected "
                    "work. A transport error is not evidence of incorrect user state."
                )
                payload["repair"] = copy.deepcopy(correction)
                if raw is not None:
                    payload["previous_output_excerpt"] = repair_excerpt(
                        raw, correction, self.settings.recovery.correction_max_characters
                    )
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            generation = self.generation(role)
            decoding = self.settings.structured_decoding.get(role, "schema")
            entry = None
            try:
                estimate, estimator = await self.preflight(role, messages, generation)
                entry = {
                    "role": role,
                    "mode": mode,
                    "attempt": 1 + sum(c["role"] == role for c in self.audit["calls"]),
                    "input_version": digest(context),
                    "input_hash": digest(messages),
                    "schema_hash": digest(schema),
                    "prompt_hash": prompt.hash,
                    "started_offset_seconds": time.perf_counter() - self.started,
                    "input_token_estimate": estimate,
                    "input_estimator": estimator,
                    "generation": generation.model_dump(mode="json"),
                    "structured_decoding": decoding,
                    "response_schema_sent": decoding == "schema",
                    "call_timeout_seconds": self.settings.recovery.call_timeout_seconds,
                    "response_received": False,
                }
                entry["artifact_id"] = f"{self.recorder.turn_id}:{role}:{entry['attempt']}"
                self.audit["calls"].append(entry)
                self.recorder.emit(
                    f"assistant_{role}_requested",
                    {
                        **entry,
                        "context": context,
                        "messages": messages,
                        "response_schema": schema,
                    },
                )
                start = time.perf_counter()
                try:
                    async with self.semaphore:
                        entry["queue_seconds"] = time.perf_counter() - start
                        async with asyncio.timeout(self.settings.recovery.call_timeout_seconds):
                            response = await self.client.generate(
                                messages=messages,
                                generation=generation,
                                response_schema=schema if decoding == "schema" else None,
                            )
                    raw = response.content
                    metadata = response.metadata or {}
                    entry.update(
                        raw_output=raw,
                        response_received=True,
                        usage=public_usage(metadata),
                        status="received",
                        wall_seconds=time.perf_counter() - start,
                    )
                    if metadata.get("transport_retry_count", 0) != 0:
                        raise ConfigurationError(
                            "GP requires one transport attempt per runtime call"
                        )
                    self.recorder.emit(f"assistant_{role}_raw_result", entry)
                    if metadata.get("finish_reason") == "length":
                        raise ContractError(
                            "output.truncated", role, "Provider reported truncation"
                        )
                    if not isinstance(raw, str) or not raw.strip():
                        raise ContractError("output.empty", role, "Empty role output")
                except BaseException as exc:
                    if entry.get("status") != "received":
                        entry.update(
                            status="cancelled"
                            if isinstance(exc, asyncio.CancelledError)
                            else "failed",
                            error_type=type(exc).__name__,
                            usage=public_usage(getattr(exc, "call_metadata", None)),
                            wall_seconds=time.perf_counter() - start,
                        )
                        # A client may raise on truncation after receiving text.
                        # Reuse only this attempt's recorded output, not an older
                        # transport attempt or a fabricated empty response.
                        attempts = entry["usage"].get("transport_attempts", [])
                        partial = attempts[-1].get("raw_output") if attempts else None
                        if isinstance(partial, str):
                            raw = partial
                            entry.update(raw_output=raw, response_received=True)
                            self.recorder.emit(
                                f"assistant_{role}_raw_result", {**entry, "status": "received"}
                            )
                        elif entry["usage"].get("finish_reason") is not None:
                            entry["response_received"] = True
                        self.recorder.emit(f"assistant_{role}_{entry['status']}", entry)
                    raise
                finally:
                    entry["wall_seconds"] = time.perf_counter() - start
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ContractError(
                        "output.parse", role, "Return exactly one JSON object"
                    ) from exc
                try:
                    validate_wire_required(data, schema, role)
                    result = model.model_validate(data)
                except ValidationError as exc:
                    errors = exc.errors(
                        include_input=False, include_context=False, include_url=False
                    )
                    detail = json.dumps([{"path": e["loc"], "type": e["type"]} for e in errors])
                    raise ContractError("output.schema", role, detail) from exc
                transformed = validate(result) if validate is not None else None
                entry.update(status="validated", structured_result=result.model_dump(mode="json"))
                self.recorder.emit(
                    f"assistant_{role}_validated",
                    {
                        "role": role,
                        "attempt": entry["attempt"],
                        "input_version": entry["input_version"],
                        "structured_result": entry["structured_result"],
                    },
                )
                return result if transformed is None else transformed
            except RecoveryExhausted:
                raise
            except (
                ContractError,
                ModelRequestError,
                InvalidModelResponseError,
                ConfigurationError,
                TimeoutError,
            ) as exc:
                error = classify_exception(exc, role)
                error.input_version = digest(context)
                if entry is not None:
                    error.artifact_id = entry["artifact_id"]
                if entry is not None:
                    entry.update(status="invalid", validation_error=error.payload())
                self.recorder.emit(
                    f"assistant_{role}_validation_failed",
                    {
                        "role": role,
                        **error.payload(),
                        "input_version": digest(context),
                    },
                )
                if error.owner != role:
                    # The consumer cannot take over an upstream owner's task.
                    raise error
                correction = self.repair(error)
                if error.code == "context.capacity":
                    self.expanded.add(role)
                if error.code == "output.truncated":
                    budget = self.settings.context.role_budgets.get(role, GPRoleBudget())
                    old = generation.max_completion_tokens
                    ceiling = budget.max_completion_tokens or old
                    self.output_limits[role] = min(
                        ceiling, math.ceil(old * self.settings.recovery.output_growth_factor)
                    )
                if error.code.startswith("call."):
                    recovery = self.settings.recovery
                    count = self.ledger.counts[(role, error.code)]
                    delay = min(
                        recovery.maximum_backoff_seconds,
                        recovery.initial_backoff_seconds * 2 ** (count - 1),
                    )
                    await asyncio.sleep(delay)
