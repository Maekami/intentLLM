from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from assistant.baselines.base import AssistantBaseline
from assistant.baselines.interactcomp_action_guard import (
    InteractCompActionGuard,
    InteractCompGuardEvaluation,
    parse_guard_decision,
)
from assistant.config import GenerationSettings
from assistant.domain.messages import ChatMessage
from assistant.exceptions import InvalidModelResponseError
from assistant.llm.base import ChatLLMClient, GeneratedResponse
from assistant.prompt import SystemPrompt
from assistant.session import AssistantSession

DEFAULT_SEMANTIC_ACTION_BUDGET_PER_TURN = 10
DEFAULT_FORMAT_RETRIES_PER_SLOT = 3


class InvalidInteractCompActionError(InvalidModelResponseError):
    """Raised when the model repeatedly fails the InteractComp action protocol."""


class InteractCompActionFormatError(InvalidInteractCompActionError):
    """Raised after format/schema retries are exhausted in one semantic slot."""


class InteractCompActionBudgetExhaustedError(InvalidInteractCompActionError):
    """Raised after every local semantic action slot is guard-rejected."""


@dataclass(frozen=True)
class InteractCompAction:
    """One validated ask/answer action selected by the adapted ReAct policy."""

    name: Literal["ask", "answer"]
    content: str
    confidence: str | None = None

    def to_payload(self) -> dict[str, Any]:
        if self.name == "ask":
            params = {"question": self.content}
        else:
            params = {"answer": self.content, "confidence": self.confidence}
        return {"action": self.name, "params": params}

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class InteractCompReActBaseline(AssistantBaseline):
    """InteractComp ask-only action policy adapted to natural dialogue.

    The reference benchmark calls this action set ``ask_only`` even though it
    contains both ask and answer. Search is deliberately absent here. Unlike
    the reference runner, answer is not a terminal action and no final-round
    prompt is installed; termination remains entirely external to the
    assistant.
    """

    name = "interactcomp_react"

    def __init__(
        self,
        prompt: SystemPrompt,
        *,
        action_guard: InteractCompActionGuard | None = None,
        semantic_action_budget_per_turn: int = DEFAULT_SEMANTIC_ACTION_BUDGET_PER_TURN,
        format_retries_per_slot: int = DEFAULT_FORMAT_RETRIES_PER_SLOT,
    ) -> None:
        if semantic_action_budget_per_turn < 1:
            raise ValueError("semantic_action_budget_per_turn must be at least 1")
        if format_retries_per_slot < 0:
            raise ValueError("format_retries_per_slot must be greater than or equal to 0")
        self.prompt = prompt
        self.action_guard = action_guard
        self.semantic_action_budget_per_turn = semantic_action_budget_per_turn
        self.format_retries_per_slot = format_retries_per_slot

    def build_messages(
        self,
        history: Sequence[ChatMessage],
        *,
        invalid_feedback: str | None = None,
    ) -> list[dict[str, Any]]:
        messages = [
            {"role": "system", "content": self.prompt.system},
            *[message.model_dump(mode="json", exclude_none=True) for message in history],
        ]
        if invalid_feedback:
            # Official ReAct retries expose invalid-action feedback as the next
            # ACT_PROMPT observation (a user message), not as system authority.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "## Observation\n"
                        f"{invalid_feedback}\n\n"
                        "## Action\n"
                        "You should output the action you want to execute.\n"
                        "Output your next action in JSON format."
                    ),
                }
            )
        return messages


class InteractCompReActSession(AssistantSession):
    """Execute one non-terminal ask/answer action for each accepted turn."""

    def __init__(
        self,
        client: ChatLLMClient,
        generation: GenerationSettings,
        baseline: AssistantBaseline,
    ) -> None:
        if not isinstance(baseline, InteractCompReActBaseline):
            raise TypeError("InteractCompReActSession requires an InteractCompReActBaseline")
        super().__init__(client, generation, baseline)
        self.react_baseline = baseline
        # The simulator and audit trail receive plain action payloads. The model
        # retains canonical action JSON, matching InteractComp's action memory.
        self._model_history: list[ChatMessage] = []
        self._last_react_call_metadata: dict[str, Any] = {}
        self._action_counts: Counter[str] = Counter()
        self._committed_invalid_action_count = 0

    @property
    def last_call_metadata(self) -> dict[str, Any]:
        return dict(self._last_react_call_metadata)

    async def respond(self, user_message: str) -> str:
        pending_user = ChatMessage(role="user", content=user_message)
        pending_model_history = [*self._model_history, pending_user]
        pending_visible_history = [*self._history, pending_user]
        all_call_metadata: list[dict[str, Any]] = []
        agent_call_metadata: list[dict[str, Any]] = []
        guard_call_metadata: list[dict[str, Any]] = []
        guard_decisions: list[dict[str, Any]] = []
        format_rejection_count = 0
        guard_rejection_count = 0
        guard_vote_count = 0
        guard_disagreement_count = 0
        guard_cache_hit_count = 0
        guard_invalid_output_count = 0
        semantic_slots_used = 0
        semantic_budget = self.react_baseline.semantic_action_budget_per_turn

        for semantic_slot in range(1, semantic_budget + 1):
            invalid_feedback: str | None = None
            last_format_error: InvalidInteractCompActionError | None = None
            action: InteractCompAction | None = None
            generated: GeneratedResponse | None = None

            # Official max_invalid_retries=3 is a same-round parser/action-name
            # retry budget. It must not consume one of the semantic slots.
            for format_attempt in range(self.react_baseline.format_retries_per_slot + 1):
                request_messages = self.react_baseline.build_messages(
                    pending_model_history,
                    invalid_feedback=invalid_feedback,
                )
                generated = await self.client.generate(
                    messages=request_messages,
                    generation=self.generation,
                )
                agent_metadata = _response_metadata(generated, self.client)
                agent_call_metadata.append(agent_metadata)
                all_call_metadata.append(agent_metadata)
                try:
                    action = parse_interactcomp_action(generated.content)
                    break
                except InvalidInteractCompActionError as exc:
                    format_rejection_count += 1
                    last_format_error = exc
                    invalid_feedback = "Invalid action. Please choose from: ask, answer."
                    if format_attempt < self.react_baseline.format_retries_per_slot:
                        continue
                    self._last_react_call_metadata = self._aggregate_metadata(
                        all_call_metadata,
                        agent_calls=agent_call_metadata,
                        guard_calls=guard_call_metadata,
                        action=None,
                        format_rejection_count=format_rejection_count,
                        guard_rejection_count=guard_rejection_count,
                        guard_decisions=guard_decisions,
                        semantic_slots_used=semantic_slots_used,
                        semantic_budget=semantic_budget,
                        budget_exhausted=False,
                        guard_vote_count=guard_vote_count,
                        guard_disagreement_count=guard_disagreement_count,
                        guard_cache_hit_count=guard_cache_hit_count,
                        guard_invalid_output_count=guard_invalid_output_count,
                    )
                    raise InteractCompActionFormatError(
                        "model failed to produce a schema-valid ask/answer action after "
                        f"{format_attempt + 1} attempts in semantic slot {semantic_slot}; "
                        f"last error: {exc}"
                    ) from exc

            if action is None or generated is None:
                raise RuntimeError(
                    f"unreachable InteractComp format retry state: {last_format_error}"
                )

            guard = self.react_baseline.action_guard
            if guard is not None:
                try:
                    evaluation = await guard.evaluate(
                        history=pending_visible_history,
                        candidate_action=action.to_payload(),
                    )
                except Exception:
                    # Transport/provider retry is owned by the injected guard
                    # client. A final failure is not an agent action rejection.
                    guard_metadata = getattr(guard.client, "last_call_metadata", {})
                    if isinstance(guard_metadata, dict) and guard_metadata:
                        guard_call_metadata.append(dict(guard_metadata))
                        all_call_metadata.append(dict(guard_metadata))
                    self._last_react_call_metadata = self._aggregate_metadata(
                        all_call_metadata,
                        agent_calls=agent_call_metadata,
                        guard_calls=guard_call_metadata,
                        action=None,
                        format_rejection_count=format_rejection_count,
                        guard_rejection_count=guard_rejection_count,
                        guard_decisions=guard_decisions,
                        semantic_slots_used=semantic_slots_used,
                        semantic_budget=semantic_budget,
                        budget_exhausted=False,
                        guard_vote_count=guard_vote_count,
                        guard_disagreement_count=guard_disagreement_count,
                        guard_cache_hit_count=guard_cache_hit_count,
                        guard_invalid_output_count=guard_invalid_output_count,
                    )
                    raise

                for response in evaluation.responses:
                    guard_metadata = _response_metadata(response, guard.client)
                    guard_call_metadata.append(guard_metadata)
                    all_call_metadata.append(guard_metadata)
                    if not parse_guard_decision(response.content).valid_output:
                        guard_invalid_output_count += 1
                if evaluation.cache_hit:
                    guard_cache_hit_count += 1
                else:
                    guard_vote_count += len(evaluation.votes)
                    if len({vote.decision.ok for vote in evaluation.votes}) > 1:
                        guard_disagreement_count += 1

                semantic_slots_used += 1
                guard_decisions.append(
                    _guard_decision_record(
                        semantic_slot=semantic_slot,
                        action=action,
                        evaluation=evaluation,
                        guard_client=guard.client,
                    )
                )
                if not evaluation.decision.ok:
                    guard_rejection_count += 1
                    # Official AskNLAction exposes only ask_invalid and records
                    # the rejected action plus observation in LinearMemory. Keep
                    # the equivalent messages in private model history; neither
                    # is published to the simulator/transcript.
                    pending_model_history.extend(
                        (
                            ChatMessage(
                                role="assistant",
                                content=action.to_json(),
                                reasoning_content=generated.reasoning_content,
                                reasoning_details=generated.reasoning_details,
                            ),
                            ChatMessage(
                                role="user",
                                content=_semantic_invalid_observation(action),
                            ),
                        )
                    )
                    if semantic_slot < semantic_budget:
                        continue
                    self._last_react_call_metadata = self._aggregate_metadata(
                        all_call_metadata,
                        agent_calls=agent_call_metadata,
                        guard_calls=guard_call_metadata,
                        action=None,
                        format_rejection_count=format_rejection_count,
                        guard_rejection_count=guard_rejection_count,
                        guard_decisions=guard_decisions,
                        semantic_slots_used=semantic_slots_used,
                        semantic_budget=semantic_budget,
                        budget_exhausted=True,
                        guard_vote_count=guard_vote_count,
                        guard_disagreement_count=guard_disagreement_count,
                        guard_cache_hit_count=guard_cache_hit_count,
                        guard_invalid_output_count=guard_invalid_output_count,
                    )
                    raise InteractCompActionBudgetExhaustedError(
                        "action guard rejected every candidate in all "
                        f"{semantic_budget} semantic action slots for one visible turn"
                    )
            else:
                # Guard-off is exactly the first-layer prompt/schema path.
                semantic_slots_used += 1

            visible_assistant = ChatMessage(
                role="assistant",
                content=action.content,
                reasoning_content=generated.reasoning_content,
                reasoning_details=generated.reasoning_details,
            )
            model_assistant = ChatMessage(
                role="assistant",
                content=action.to_json(),
                reasoning_content=generated.reasoning_content,
                reasoning_details=generated.reasoning_details,
            )
            self._history.extend((pending_user, visible_assistant))
            # Commit the pending user, any private invalid action/observation
            # pairs, and the accepted canonical action atomically.
            self._model_history = [*pending_model_history, model_assistant]
            self._action_counts[action.name] += 1
            self._last_react_call_metadata = self._aggregate_metadata(
                all_call_metadata,
                agent_calls=agent_call_metadata,
                guard_calls=guard_call_metadata,
                action=action,
                format_rejection_count=format_rejection_count,
                guard_rejection_count=guard_rejection_count,
                guard_decisions=guard_decisions,
                semantic_slots_used=semantic_slots_used,
                semantic_budget=semantic_budget,
                budget_exhausted=False,
                guard_vote_count=guard_vote_count,
                guard_disagreement_count=guard_disagreement_count,
                guard_cache_hit_count=guard_cache_hit_count,
                guard_invalid_output_count=guard_invalid_output_count,
            )
            self._committed_invalid_action_count += guard_rejection_count
            # Both ask and answer return one visible non-terminal message. END
            # and the external 20-turn limit remain entirely system-controlled.
            return action.content

        raise RuntimeError("unreachable InteractComp semantic action state")

    def _aggregate_metadata(
        self,
        calls: list[dict[str, Any]],
        *,
        agent_calls: list[dict[str, Any]],
        guard_calls: list[dict[str, Any]],
        action: InteractCompAction | None,
        format_rejection_count: int,
        guard_rejection_count: int,
        guard_decisions: list[dict[str, Any]],
        semantic_slots_used: int,
        semantic_budget: int,
        budget_exhausted: bool,
        guard_vote_count: int,
        guard_disagreement_count: int,
        guard_cache_hit_count: int,
        guard_invalid_output_count: int,
    ) -> dict[str, Any]:
        return _aggregate_call_metadata(
            calls,
            agent_calls=agent_calls,
            guard_calls=guard_calls,
            action=action,
            format_rejection_count=format_rejection_count,
            guard_rejection_count=guard_rejection_count,
            guard_enabled=self.react_baseline.action_guard is not None,
            guard_decisions=guard_decisions,
            cumulative_action_counts=self._action_counts,
            semantic_slots_used=semantic_slots_used,
            semantic_budget=semantic_budget,
            format_retries_per_slot=self.react_baseline.format_retries_per_slot,
            budget_exhausted=budget_exhausted,
            guard_vote_count=guard_vote_count,
            guard_disagreement_count=guard_disagreement_count,
            guard_cache_hit_count=guard_cache_hit_count,
            guard_invalid_output_count=guard_invalid_output_count,
            private_invalid_action_count=(
                self._committed_invalid_action_count + guard_rejection_count
            ),
        )

    def replace_history(self, messages: Sequence[ChatMessage]) -> None:
        # No caller currently rehydrates these sessions. If one does, visible
        # messages are the safest lossless fallback because historical action
        # labels cannot be reconstructed unambiguously from plain text.
        super().replace_history(messages)
        self._model_history = list(messages)
        self._last_react_call_metadata = {}
        self._action_counts.clear()
        self._committed_invalid_action_count = 0
        if self.react_baseline.action_guard is not None:
            self.react_baseline.action_guard.reset_cache()

    def reset(self) -> None:
        super().reset()
        self._model_history.clear()
        self._last_react_call_metadata = {}
        self._action_counts.clear()
        self._committed_invalid_action_count = 0
        if self.react_baseline.action_guard is not None:
            self.react_baseline.action_guard.reset_cache()


def parse_interactcomp_action(response: str) -> InteractCompAction:
    """Parse and validate the reference ``{action, params}`` JSON protocol."""

    data = _extract_json_object(response)
    action_name = data.get("action") or data.get("name") or ""
    if not isinstance(action_name, str):
        raise InvalidInteractCompActionError("action must be a string")
    action_name = action_name.strip()
    if action_name not in {"ask", "answer"}:
        raise InvalidInteractCompActionError("invalid action; choose exactly one of: ask, answer")

    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise InvalidInteractCompActionError("params must be a JSON object")

    if action_name == "ask":
        _require_exact_params(params, {"question"}, action_name)
        question = params.get("question")
        if not isinstance(question, str) or not question.strip():
            raise InvalidInteractCompActionError("ask requires a non-empty string params.question")
        return InteractCompAction(name="ask", content=question.strip())

    _require_exact_params(params, {"answer", "confidence"}, action_name)
    answer = params.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise InvalidInteractCompActionError("answer requires a non-empty string params.answer")
    confidence = params.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (str, int, float)):
        raise InvalidInteractCompActionError(
            "answer requires scalar params.confidence in the 0-100 convention"
        )
    confidence_text = str(confidence).strip()
    if not confidence_text:
        raise InvalidInteractCompActionError("answer requires non-empty params.confidence")
    return InteractCompAction(
        name="answer",
        content=answer.strip(),
        confidence=confidence_text,
    )


def _extract_json_object(response: str) -> dict[str, Any]:
    text = (response or "").strip()
    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*([\s\S]*?)```",
            text,
            flags=re.IGNORECASE,
        )
    )
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    raise InvalidInteractCompActionError("response must contain one valid JSON action object")


def _require_exact_params(
    params: dict[str, Any],
    expected: set[str],
    action_name: str,
) -> None:
    actual = set(params)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unknown:
        details.append(f"unexpected {unknown}")
    raise InvalidInteractCompActionError(f"{action_name} params are invalid ({'; '.join(details)})")


def _semantic_invalid_observation(action: InteractCompAction) -> str:
    """Return the private official-style observation for one rejected action."""

    return (
        "## Observation\n"
        f"Last action: {action.name}\n"
        f"Action: {action.to_json()}\n"
        f"Observation: {action.name}_invalid\n\n"
        "## Action\n"
        "You should output the action you want to execute.\n"
        "Output your next action in JSON format."
    )


def _validator_attempt_records(
    responses: Sequence[GeneratedResponse],
    client: ChatLLMClient,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, response in enumerate(responses, start=1):
        decision = parse_guard_decision(response.content)
        metadata = _response_metadata(response, client)
        records.append(
            {
                "attempt": index,
                "valid_output": decision.valid_output,
                "ok": decision.ok if decision.valid_output else None,
                "reason": decision.reason,
                "request_id": metadata.get("request_id"),
                "validator_response": response.content,
            }
        )
    return records


def _guard_decision_record(
    *,
    semantic_slot: int,
    action: InteractCompAction,
    evaluation: InteractCompGuardEvaluation,
    guard_client: ChatLLMClient,
) -> dict[str, Any]:
    votes: list[dict[str, Any]] = []
    for index, vote in enumerate(evaluation.votes, start=1):
        metadata = _response_metadata(vote.response, guard_client)
        votes.append(
            {
                "vote": index,
                "ok": vote.decision.ok,
                "reason": vote.decision.reason,
                "request_id": metadata.get("request_id"),
                "validator_response": vote.response.content,
            }
        )
    return {
        "semantic_slot": semantic_slot,
        "action": action.name,
        "candidate_action": action.to_payload(),
        "status": "accepted" if evaluation.decision.ok else "rejected",
        "ok": evaluation.decision.ok,
        "reason": evaluation.decision.reason,
        "cache_hit": evaluation.cache_hit,
        "cache_key": evaluation.cache_key,
        "validator_attempts": _validator_attempt_records(
            evaluation.responses,
            guard_client,
        ),
        "votes": votes,
    }


def _response_metadata(
    generated: GeneratedResponse,
    client: ChatLLMClient,
) -> dict[str, Any]:
    if isinstance(generated.metadata, dict):
        return dict(generated.metadata)
    metadata = getattr(client, "last_call_metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _aggregate_call_metadata(
    calls: list[dict[str, Any]],
    *,
    agent_calls: list[dict[str, Any]],
    guard_calls: list[dict[str, Any]],
    action: InteractCompAction | None,
    format_rejection_count: int,
    guard_rejection_count: int,
    guard_enabled: bool,
    guard_decisions: list[dict[str, Any]],
    cumulative_action_counts: Counter[str],
    semantic_slots_used: int,
    semantic_budget: int,
    format_retries_per_slot: int,
    budget_exhausted: bool,
    guard_vote_count: int,
    guard_disagreement_count: int,
    guard_cache_hit_count: int,
    guard_invalid_output_count: int,
    private_invalid_action_count: int,
) -> dict[str, Any]:
    # Preserve the accepted/final candidate model call as the primary response
    # metadata even though a successful guard call occurs after it.
    result = dict(agent_calls[-1]) if agent_calls else {}
    result.update(
        {
            "latency_seconds": _sum_numbers(calls, "latency_seconds"),
            "transport_retry_count": _sum_integers(calls, "transport_retry_count"),
            "input_tokens": _sum_integers(calls, "input_tokens", strict=True),
            "output_tokens": _sum_integers(calls, "output_tokens", strict=True),
            "thinking_tokens": _sum_integers(calls, "thinking_tokens", strict=True),
            "answer_tokens": _sum_integers(calls, "answer_tokens", strict=True),
            "reasoning_preserved": any(call.get("reasoning_preserved") is True for call in calls),
            "internal_call_count": len(calls),
            "interactcomp_agent_call_count": len(agent_calls),
            # One semantic action candidate can require multiple same-slot
            # format calls, so these counters must remain distinct.
            "interactcomp_action_attempt_count": semantic_slots_used,
            "interactcomp_semantic_action_slots_used": semantic_slots_used,
            "interactcomp_semantic_action_budget_per_turn": semantic_budget,
            "interactcomp_semantic_action_budget_exhausted": budget_exhausted,
            "interactcomp_format_retries_per_slot": format_retries_per_slot,
            "interactcomp_action": action.name if action is not None else None,
            "interactcomp_confidence": (action.confidence if action is not None else None),
            "interactcomp_invalid_action_count": (format_rejection_count + guard_rejection_count),
            "interactcomp_format_rejection_count": format_rejection_count,
            "interactcomp_guard_enabled": guard_enabled,
            "interactcomp_guard_call_count": len(guard_calls),
            "interactcomp_guard_rejection_count": guard_rejection_count,
            "interactcomp_guard_vote_count": guard_vote_count,
            "interactcomp_guard_disagreement_count": guard_disagreement_count,
            "interactcomp_guard_cache_hit_count": guard_cache_hit_count,
            "interactcomp_guard_invalid_output_count": guard_invalid_output_count,
            "interactcomp_guard_decisions": list(guard_decisions),
            "interactcomp_private_invalid_action_memory_count": (private_invalid_action_count),
            "interactcomp_cumulative_action_counts": dict(cumulative_action_counts),
            "interactcomp_answer_is_terminal": False,
        }
    )
    request_ids = [call.get("request_id") for call in calls if call.get("request_id")]
    if len(request_ids) > 1:
        result["request_ids"] = request_ids
    result["interactcomp_agent_request_ids"] = [
        call.get("request_id") for call in agent_calls if call.get("request_id")
    ]
    result["interactcomp_guard_request_ids"] = [
        call.get("request_id") for call in guard_calls if call.get("request_id")
    ]
    return result


def _sum_integers(
    calls: list[dict[str, Any]],
    key: str,
    *,
    strict: bool = False,
) -> int | None:
    values = [call.get(key) for call in calls]
    valid = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
    if strict and len(valid) != len(values):
        return None
    return sum(valid) if valid else (None if strict else 0)


def _sum_numbers(calls: list[dict[str, Any]], key: str) -> float:
    return sum(
        float(value)
        for call in calls
        if isinstance((value := call.get(key)), (int, float)) and not isinstance(value, bool)
    )
