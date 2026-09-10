from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from assistant.baselines.base import AssistantBaseline
from assistant.config import GenerationSettings, MemorySettings
from assistant.domain.messages import ChatMessage
from assistant.llm.base import ChatLLMClient, GeneratedResponse
from assistant.memory.context import (
    REMEM_SYSTEM_PROMPT,
    add_system_prompt,
    build_exprag_context,
    build_remem_context,
    replace_latest_user_content,
)
from assistant.memory.models import MemoryEntry, PreparedMemoryUpdate, RetrievalResult
from assistant.memory.retrieval import MemoryRetriever
from assistant.memory.store import JsonMemoryStore
from assistant.session import AssistantSession


class EvolvingMemorySession(AssistantSession):
    """Shared Search-Synthesize-Evolve lifecycle for one pipeline episode."""

    framework: str

    def __init__(
        self,
        client: ChatLLMClient,
        generation: GenerationSettings,
        baseline: AssistantBaseline,
        *,
        settings: MemorySettings,
        store: JsonMemoryStore,
        retriever: MemoryRetriever,
        memory_snapshot: Sequence[MemoryEntry] | None = None,
    ) -> None:
        super().__init__(client, generation, baseline)
        if settings.framework != self.framework:
            raise ValueError(
                f"{type(self).__name__} requires framework={self.framework!r}, "
                f"not {settings.framework!r}"
            )
        self.memory_settings = settings
        self.memory_store = store
        self.retriever = retriever
        self._memory_snapshot = tuple(memory_snapshot) if memory_snapshot is not None else None
        self._episode_memory_snapshot: tuple[MemoryEntry, ...] | None = self._memory_snapshot
        self._retrieved: list[RetrievalResult] = []
        self._task_input: str | None = None
        self._last_retrieval_query = ""
        self._task_finalized = False
        self._last_memory_call_metadata: dict[str, Any] = {}

    @property
    def memory_framework(self) -> str:
        return self.framework

    @property
    def retrieved(self) -> tuple[RetrievalResult, ...]:
        return tuple(self._retrieved)

    @property
    def last_call_metadata(self) -> dict[str, Any]:
        if self._last_memory_call_metadata:
            return dict(self._last_memory_call_metadata)
        return super().last_call_metadata

    def _begin_task(self, query: str) -> None:
        if self._task_input is None:
            self._task_input = query
        if self._episode_memory_snapshot is None:
            # Freeze the bank for the whole episode. Evolution runs provide an
            # explicit mini-batch snapshot; ordinary sessions lazily freeze the
            # store as it exists when the episode begins.
            self._episode_memory_snapshot = tuple(self.memory_store.entries())

    def _retrieve_for_turn(self, pending_history: Sequence[ChatMessage]) -> None:
        if self._episode_memory_snapshot is None:
            raise RuntimeError("memory bank was not initialized for the current task")
        retrieval = self.memory_settings.retrieval
        query = _build_retrieval_query(
            pending_history,
            maximum=retrieval.query_max_characters,
        )
        self._last_retrieval_query = query
        self._retrieved = self.retriever.retrieve(
            query,
            list(self._episode_memory_snapshot),
            top_k=retrieval.top_k,
            min_score=retrieval.min_score,
        )

    def _prepare_turn(self, user_message: str) -> tuple[ChatMessage, list[ChatMessage]]:
        if self._task_finalized:
            raise RuntimeError(
                "memory task is finalized; call reset() before starting another task"
            )
        self._begin_task(user_message)
        pending_user = ChatMessage(role="user", content=user_message)
        pending_history = [*self._history, pending_user]
        self._retrieve_for_turn(pending_history)
        return pending_user, pending_history

    def _retrieval_metadata(
        self,
        retrieved: Sequence[RetrievalResult] | None = None,
    ) -> dict[str, Any]:
        results = list(self._retrieved if retrieved is None else retrieved)
        query = self._last_retrieval_query
        return {
            "memory_framework": self.framework,
            "memory_retrieval_backend": self.memory_settings.retrieval.backend,
            "memory_retrieval_scope": "per_turn",
            "memory_bank_scope": "frozen_episode_snapshot",
            "memory_bank_snapshot_source": (
                "provided_snapshot"
                if self._memory_snapshot is not None
                else "store_at_episode_start"
            ),
            "memory_bank_entry_count": len(self._episode_memory_snapshot or ()),
            "memory_retrieval_query_strategy": "visible_user_turns_latest_first",
            "memory_retrieval_query_character_count": len(query),
            "memory_retrieval_query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "memory_retrieved_count": len(results),
            "memory_retrieved_task_ids": [result.entry.task_id for result in results],
            "memory_retrieval_scores": [result.score for result in results],
        }

    def _commit(
        self,
        pending_user: ChatMessage,
        generated: GeneratedResponse,
        *,
        visible_content: str | None = None,
    ) -> str:
        content = visible_content if visible_content is not None else generated.content
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
            reasoning_content=generated.reasoning_content,
            reasoning_details=generated.reasoning_details,
        )
        self._history.extend((pending_user, assistant_message))
        return content

    def _record_metadata(self, metadata: dict[str, Any]) -> None:
        self._last_memory_call_metadata = metadata
        # Preserve the historical public accounting location used by callers
        # and by older interaction_pipeline versions.
        try:
            self.client.last_call_metadata = dict(metadata)
        except (AttributeError, TypeError):
            pass

    def finalize_task(
        self,
        *,
        task_id: str | None = None,
        success: bool,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """Evolve memory immediately after an externally evaluated episode ends."""

        prepared = self.prepare_memory_update(
            task_id=task_id,
            success=success,
            feedback=feedback,
        )
        if prepared.entry is None:
            return self.memory_store.skipped(
                task_id=prepared.task_id,
                reason=prepared.reason,
            ).to_dict()
        return self.memory_store.upsert(prepared.entry).to_dict()

    def prepare_memory_update(
        self,
        *,
        task_id: str | None = None,
        success: bool,
        feedback: str | None = None,
    ) -> PreparedMemoryUpdate:
        """Finalize one episode without persisting it, for mini-batch evolution."""

        if self._task_finalized:
            return PreparedMemoryUpdate(
                task_id=task_id,
                reason="task_already_finalized",
                entry=None,
            )
        self._task_finalized = True
        if not self._history:
            return PreparedMemoryUpdate(
                task_id=task_id,
                reason="no_completed_assistant_turn",
                entry=None,
            )
        resolved_task_id = task_id or _content_task_id(self._history)
        if self.memory_settings.store_successful_only and not success:
            return PreparedMemoryUpdate(
                task_id=resolved_task_id,
                reason="unsuccessful_task_not_stored",
                entry=None,
            )
        entry = MemoryEntry(
            task_id=resolved_task_id,
            input_text=self._task_input or _first_user_content(self._history),
            output_text=_last_assistant_content(self._history),
            feedback=feedback,
            trajectory=_conversation_trajectory(self._history),
            metadata={
                "framework": self.framework,
                "underlying_baseline": self.baseline.name,
                "model_profile": getattr(
                    getattr(self.client, "profile", None),
                    "profile_name",
                    None,
                ),
            },
            is_successful=success,
        )
        return PreparedMemoryUpdate(
            task_id=resolved_task_id,
            entry=entry,
            reason="ready_for_batch_commit",
        )

    def replace_history(self, messages: Sequence[ChatMessage]) -> None:
        super().replace_history(messages)
        self._task_input = _first_user_content(self._history) if self._history else None
        self._episode_memory_snapshot = self._memory_snapshot
        self._retrieved = []
        self._last_retrieval_query = ""
        self._task_finalized = False
        self._last_memory_call_metadata = {}
        if self._task_input is not None:
            self._begin_task(self._task_input)

    def reset(self) -> None:
        super().reset()
        self._episode_memory_snapshot = self._memory_snapshot
        self._retrieved = []
        self._task_input = None
        self._last_retrieval_query = ""
        self._task_finalized = False
        self._last_memory_call_metadata = {}


class ExpRAGSession(EvolvingMemorySession):
    """Experience retrieval and in-context aggregation baseline."""

    framework = "exprag"

    async def respond(self, user_message: str) -> str:
        pending_user, pending_history = self._prepare_turn(user_message)
        request_messages = self.baseline.build_messages(pending_history)
        context = build_exprag_context(
            user_message,
            self._retrieved,
            self.memory_settings.context,
        )
        request_messages = replace_latest_user_content(request_messages, context)
        generated = await self.client.generate(
            messages=request_messages,
            generation=self.generation,
        )
        metadata = dict(getattr(self.client, "last_call_metadata", {}) or {})
        metadata.update(self._memory_metadata())
        self._record_metadata(metadata)
        return self._commit(pending_user, generated)

    def _memory_metadata(self) -> dict[str, Any]:
        return self._retrieval_metadata()


class ReMemSession(EvolvingMemorySession):
    """ReMem's model-directed Think-Prune-Final Answer decision loop."""

    framework = "remem"

    async def respond(self, user_message: str) -> str:
        pending_user, pending_history = self._prepare_turn(user_message)
        initially_retrieved = tuple(self._retrieved)
        self._last_memory_call_metadata = {}
        reasoning_trace: list[str] = []
        actions: list[str] = []
        call_metadata: list[dict[str, Any]] = []
        last_generated: GeneratedResponse | None = None
        visible_content: str | None = None

        for _ in range(self.memory_settings.remem.max_iterations):
            request_messages = self.baseline.build_messages(pending_history)
            request_messages = add_system_prompt(request_messages, REMEM_SYSTEM_PROMPT)
            context = build_remem_context(
                user_message,
                self._retrieved,
                reasoning_trace,
                self.memory_settings.context,
            )
            request_messages = replace_latest_user_content(request_messages, context)
            last_generated = await self.client.generate(
                messages=request_messages,
                generation=self.generation,
            )
            call_metadata.append(dict(getattr(self.client, "last_call_metadata", {}) or {}))
            action, content = _parse_remem_action(last_generated.content)
            actions.append(action)
            if action == "think":
                reasoning_trace.append(f"Think: {content}")
                continue
            if action == "refine":
                if self.memory_settings.remem.enable_pruning:
                    removed = self._prune(content)
                    suffix = f" (removed {removed})" if removed else " (no valid IDs)"
                else:
                    suffix = " (pruning disabled)"
                reasoning_trace.append(f"Refine: {content}{suffix}")
                continue
            visible_content = content
            break

        if last_generated is None:  # guarded by config validation, retained defensively
            raise RuntimeError("ReMem made no model call")
        if visible_content is None:
            visible_content = _extract_answer(last_generated.content)
        aggregate = _aggregate_metadata(call_metadata, actions)
        aggregate.update(self._retrieval_metadata(initially_retrieved))
        retained_task_ids = [result.entry.task_id for result in self._retrieved]
        retained_id_set = set(retained_task_ids)
        aggregate.update(
            {
                "memory_retained_count": len(self._retrieved),
                "memory_retained_task_ids": retained_task_ids,
                "memory_retained_scores": [result.score for result in self._retrieved],
                "memory_pruned_count": len(initially_retrieved) - len(self._retrieved),
                "memory_pruned_task_ids": [
                    result.entry.task_id
                    for result in initially_retrieved
                    if result.entry.task_id not in retained_id_set
                ],
            }
        )
        self._record_metadata(aggregate)
        return self._commit(
            pending_user,
            last_generated,
            visible_content=visible_content,
        )

    def _prune(self, value: str) -> int:
        indexes = _parse_prune_indexes(value, len(self._retrieved))
        if not indexes:
            return 0
        self._retrieved = [
            result for index, result in enumerate(self._retrieved, 1) if index not in indexes
        ]
        return len(indexes)


def _parse_remem_action(response: str) -> tuple[str, str]:
    value = response.strip()
    # Gemini may copy the Markdown list marker used to describe an operation.
    # Normalize only a marker directly preceding a known ReMem control label so
    # ordinary user-visible answers that begin with a list remain untouched.
    value = re.sub(
        r"^[-*+]\s+(?=(?:Think-Prune|Think|Final Answer|Action)\s*:)",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    patterns = (
        # Match only the prune payload's line, as in the official ReMem parser.
        # Otherwise a following Final Answer block is swallowed into the ID list.
        ("refine", r"Think-Prune:[^\S\r\n]*([^\r\n]+)", re.IGNORECASE),
        ("think", r"Think:\s*(.+)", re.IGNORECASE | re.DOTALL),
        ("act", r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL),
        ("act", r"Action:\s*(.+)", re.IGNORECASE | re.DOTALL),
    )
    for action, pattern, flags in patterns:
        match = re.match(pattern, value, flags)
        if match:
            return action, match.group(1).strip()
    return "act", value


def _parse_prune_indexes(value: str, maximum: int) -> set[int]:
    indexes: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            pieces = part.split("-", 1)
            try:
                start, end = map(int, pieces)
            except ValueError:
                continue
            indexes.update(index for index in range(start, end + 1) if 1 <= index <= maximum)
        else:
            try:
                index = int(part)
            except ValueError:
                continue
            if 1 <= index <= maximum:
                indexes.add(index)
    return indexes


def _extract_answer(response: str) -> str:
    match = re.search(r"Final Answer:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def _aggregate_metadata(
    calls: list[dict[str, Any]],
    actions: list[str],
) -> dict[str, Any]:
    last = calls[-1] if calls else {}
    result = {
        key: last.get(key)
        for key in (
            "model_id",
            "model_profile",
            "provider",
            "request_id",
            "finish_reason",
        )
        if key in last
    }
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
            "remem_operation_counts": dict(Counter(actions)),
        }
    )
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


def _build_retrieval_query(
    messages: Sequence[ChatMessage],
    *,
    maximum: int,
) -> str:
    """Build a no-leakage query from user-visible intent accumulated so far.

    The latest request is placed first so embedding-model truncation cannot
    discard the newest constraint. Earlier user turns provide disambiguating
    context without echoing long assistant answers into the retrieval query.
    """

    user_turns = [
        message.content.strip()
        for message in messages
        if message.role == "user" and message.content.strip()
    ]
    if not user_turns:
        return ""
    if len(user_turns) == 1:
        return user_turns[0][:maximum]
    query = (
        "CURRENT USER REQUEST:\n"
        + user_turns[-1]
        + "\n\nEARLIER USER CONTEXT (MOST RECENT FIRST):\n"
        + "\n\n".join(reversed(user_turns[:-1]))
    )
    return query[:maximum]


def _first_user_content(messages: Sequence[ChatMessage]) -> str:
    return next((message.content for message in messages if message.role == "user"), "")


def _last_assistant_content(messages: Sequence[ChatMessage]) -> str:
    return next(
        (message.content for message in reversed(messages) if message.role == "assistant"),
        "",
    )


def _conversation_trajectory(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    trajectory: list[dict[str, str]] = []
    pending_user: str | None = None
    for message in messages:
        if message.role == "user":
            pending_user = message.content
        elif pending_user is not None:
            trajectory.append({"user": pending_user, "assistant": message.content})
            pending_user = None
    return trajectory


def _content_task_id(messages: Sequence[ChatMessage]) -> str:
    content = "\0".join(f"{message.role}:{message.content}" for message in messages)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
