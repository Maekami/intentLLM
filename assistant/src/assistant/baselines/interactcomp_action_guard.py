from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from assistant.config import GenerationSettings
from assistant.domain.messages import ChatMessage
from assistant.llm.base import ChatLLMClient, GeneratedResponse
from assistant.prompt import SystemPrompt


@dataclass(frozen=True)
class InteractCompGuardDecision:
    """One official-style ``{ok, reason}`` semantic boundary decision."""

    ok: bool
    reason: str
    valid_output: bool = True


@dataclass(frozen=True)
class InteractCompGuardVote:
    """One effective validator verdict and the call that produced it."""

    decision: InteractCompGuardDecision
    response: GeneratedResponse


@dataclass(frozen=True)
class InteractCompGuardEvaluation:
    decision: InteractCompGuardDecision
    votes: tuple[InteractCompGuardVote, ...]
    # Only calls made by this evaluation are returned. Cache hits deliberately
    # return an empty tuple so usage is not counted twice.
    responses: tuple[GeneratedResponse, ...]
    cache_hit: bool
    cache_key: str


class InteractCompActionGuard:
    """LLM validator for the adapted ask/answer communication boundary.

    InteractComp's AskNL validator uses its responder/user LLM rather than the
    tested agent LLM and expects an ``{ok, reason}`` object for the candidate
    question alone. The pipeline injects that separate simulator-profile
    client. This adaptation keeps the official contract for ``ask`` and checks
    the project's non-terminal ``answer`` boundary with visible dialogue only
    where identifying the intended respondent requires it.
    """

    def __init__(
        self,
        prompt: SystemPrompt,
        generation: GenerationSettings,
        client: ChatLLMClient,
        *,
        confirm_rejections: bool = False,
        cache_verdicts: bool = False,
        invalid_output_retries: int = 0,
    ) -> None:
        if invalid_output_retries < 0:
            raise ValueError("invalid_output_retries must be greater than or equal to 0")
        self.prompt = prompt
        self.generation = generation
        self.client = client
        self.confirm_rejections = confirm_rejections
        self.cache_verdicts = cache_verdicts
        self.invalid_output_retries = invalid_output_retries
        # A guard instance belongs to one assistant session/episode, so this is
        # an episode-local cache and cannot leak verdicts across samples.
        self._cache: dict[str, InteractCompGuardEvaluation] = {}

    async def evaluate(
        self,
        *,
        history: Sequence[ChatMessage],
        candidate_action: dict[str, Any],
    ) -> InteractCompGuardEvaluation:
        payload = _guard_payload(history=history, candidate_action=candidate_action)
        cache_key = self._cache_key(payload)
        cached = self._cache.get(cache_key) if self.cache_verdicts else None
        if cached is not None:
            return InteractCompGuardEvaluation(
                decision=cached.decision,
                votes=cached.votes,
                responses=(),
                cache_hit=True,
                cache_key=cache_key,
            )

        responses: list[GeneratedResponse] = []
        votes: list[InteractCompGuardVote] = []
        first, first_responses = await self._request_vote(payload)
        responses.extend(first_responses)
        votes.append(first)
        final_decision = first.decision

        # Confirm only a first rejection. Accepted boundaries were stable in
        # audit, while unconditional voting would triple every guard call. This
        # remains an explicit compatibility option; the official-aligned
        # baseline disables it and uses exactly one verdict per candidate.
        if self.confirm_rejections and not first.decision.ok:
            second, second_responses = await self._request_vote(payload)
            responses.extend(second_responses)
            votes.append(second)
            if not second.decision.ok:
                final_decision = first.decision
            else:
                third, third_responses = await self._request_vote(payload)
                responses.extend(third_responses)
                votes.append(third)
                false_votes = [vote for vote in votes if not vote.decision.ok]
                true_votes = [vote for vote in votes if vote.decision.ok]
                majority = false_votes if len(false_votes) > len(true_votes) else true_votes
                final_decision = majority[-1].decision

        evaluation = InteractCompGuardEvaluation(
            decision=final_decision,
            votes=tuple(votes),
            responses=tuple(responses),
            cache_hit=False,
            cache_key=cache_key,
        )
        if self.cache_verdicts:
            self._cache[cache_key] = evaluation
        return evaluation

    async def _request_vote(
        self,
        payload: dict[str, Any],
    ) -> tuple[InteractCompGuardVote, tuple[GeneratedResponse, ...]]:
        responses: list[GeneratedResponse] = []
        last_decision: InteractCompGuardDecision | None = None
        last_response: GeneratedResponse | None = None
        for attempt in range(self.invalid_output_retries + 1):
            response = await self.client.generate(
                messages=[
                    {"role": "system", "content": self.prompt.system},
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                generation=self.generation,
            )
            responses.append(response)
            decision = parse_guard_decision(response.content)
            last_decision = decision
            last_response = response
            if decision.valid_output:
                return InteractCompGuardVote(decision=decision, response=response), tuple(responses)
            if attempt >= self.invalid_output_retries:
                break

        if last_decision is None or last_response is None:
            raise RuntimeError("unreachable InteractComp guard request state")
        # Official AskNLValidator fails closed: a malformed/unparseable
        # validator result becomes ask_invalid rather than aborting the run.
        # Preserve that behavior for the adapted ask/answer boundary. With the
        # shipped invalid_output_retries=0 this is exactly one model call and
        # one rejected semantic action slot.
        return InteractCompGuardVote(
            decision=last_decision,
            response=last_response,
        ), tuple(responses)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        generation = self.generation.model_dump_json(exclude_none=True)
        return sha256(f"{self.prompt.hash}\0{generation}\0{canonical}".encode()).hexdigest()

    def reset_cache(self) -> None:
        self._cache.clear()


def _guard_payload(
    *,
    history: Sequence[ChatMessage],
    candidate_action: dict[str, Any],
) -> dict[str, Any]:
    if candidate_action.get("action") == "ask":
        params = candidate_action.get("params")
        question = params.get("question", "") if isinstance(params, dict) else ""
        # Match official AskNLValidator authority: question only, without
        # dialogue that could invite relevance or progress judgments.
        return {"question": question}
    # Answer is non-terminal in this project, so dialogue is necessary only to
    # identify whether question-shaped content solicits the current user.
    return {
        "dialogue": [{"role": message.role, "content": message.content} for message in history],
        "candidate_action": candidate_action,
    }


def parse_guard_decision(response: str) -> InteractCompGuardDecision:
    """Parse the official ``{ok, reason}`` shape and flag malformed output."""

    data = _extract_json_object(response)
    if data is None or not isinstance(data.get("ok"), bool):
        return InteractCompGuardDecision(
            ok=False,
            reason="invalid validator output: expected JSON boolean field 'ok'",
            valid_output=False,
        )
    ok = data["ok"]
    reason_value = data.get("reason")
    reason = str(reason_value).strip() if reason_value is not None else ""
    if not reason:
        reason = "non-compliant question"
    # The full response is retained in audit metadata; keep the parsed reason
    # bounded even if a validator ignores the prompt.
    return InteractCompGuardDecision(ok=ok, reason=reason[:500])


def _extract_json_object(response: str) -> dict[str, Any] | None:
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
    return None
