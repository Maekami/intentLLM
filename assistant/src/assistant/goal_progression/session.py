"""GP v2 transactional session, compatible with the existing baseline/audit entry point."""

import asyncio
import copy
import time
import uuid

from assistant.baselines.base import BaseBaseline
from assistant.domain.messages import ChatMessage
from assistant.exceptions import ConfigurationError, ModelRequestError
from assistant.session import AssistantSession

from .engine import TurnEngine
from .errors import ContractError, RecoveryExhausted, RetryLedger
from .events import AuditRecorder, EvidenceStore, IdentityIndex, InteractionIndex, digest
from .prompts import load_prompts
from .rendering import validate_receipt
from .runtime import ContractRuntime, provider_limiter, public_usage
from .schemas import SemanticSnapshot

__all__ = ["GoalProgressionBaseline", "GoalProgressionSession", "provider_limiter", "public_usage"]


class GoalProgressionBaseline(BaseBaseline):
    name = "goal_progression"

    def build_messages(self, history):
        raise ConfigurationError("goal_progression requires GoalProgressionSession")


class GoalProgressionSession(AssistantSession):
    def __init__(self, client, generation, baseline, *, profile):
        super().__init__(client, generation, baseline)
        if profile.goal_progression is None:
            raise ConfigurationError("missing goal_progression settings")
        if profile.retry.max_attempts != 1:
            raise ConfigurationError(
                "GP v2 requires retry.max_attempts=1; recovery is owned by goal_progression.recovery"
            )
        client_profile = getattr(client, "profile", None)
        if client_profile is not None and client_profile.retry.max_attempts != 1:
            raise ConfigurationError("Injected GP client has nested transport retries enabled")
        self.profile, self.settings = profile, profile.goal_progression
        self.prompts = load_prompts(profile)
        self._lock = asyncio.Lock()
        self._state = None
        self._version = 0
        self._turn_counter = 0
        self._events = []
        self._registry = IdentityIndex()
        self._interactions = InteractionIndex()
        self._bridge = None
        self._metadata = {}
        self._episode_id = uuid.uuid4().hex

    @property
    def state(self):
        return self._state.model_copy(deep=True) if self._state is not None else None

    @property
    def interaction_state(self):
        return copy.deepcopy(self._interactions.requests)

    @property
    def last_call_metadata(self):
        return copy.deepcopy(self._metadata)

    def reset(self):
        if self._lock.locked():
            raise RuntimeError("cannot reset during respond")
        AssistantSession.reset(self)
        self._state, self._bridge = None, None
        self._version, self._turn_counter = 0, 0
        self._events, self._metadata = [], {}
        self._registry, self._interactions = IdentityIndex(), InteractionIndex()
        self._episode_id = uuid.uuid4().hex

    def replace_history(self, messages):
        if self._lock.locked():
            raise RuntimeError("cannot replace history during respond")
        self.reset()
        self._history = [ChatMessage(role=m.role, content=m.content) for m in messages]
        self._events = [
            {"event_id": f"m{i + 1}", "role": m.role, "content": m.content}
            for i, m in enumerate(self._history)
        ]
        self._turn_counter = sum(m.role == "user" for m in self._history)
        self._interactions.completeness = "visible_only"
        self._interactions.imported_event_count = len(self._events)

    def export_checkpoint(self):
        if self._lock.locked():
            raise RuntimeError("cannot export an in-flight transaction")
        value = {
            "architecture_version": "v2_contracts",
            "variant": self.settings.variant,
            "episode_id": self._episode_id,
            "visible_events": copy.deepcopy(self._events),
            "snapshot": self._state.model_dump() if self._state else None,
            "version": self._version,
            "turn_counter": self._turn_counter,
            "identities": copy.deepcopy(self._registry.entries),
            "interaction": {
                "requests": copy.deepcopy(self._interactions.requests),
                "completeness": self._interactions.completeness,
                "imported_event_count": self._interactions.imported_event_count,
            },
            "bridge": copy.deepcopy(self._bridge),
        }
        return {**value, "checksum": digest(value)}

    def restore_checkpoint(self, value):
        if self._lock.locked():
            raise RuntimeError("cannot restore during respond")
        data = copy.deepcopy(value)
        checksum = data.pop("checksum", None)
        if (
            checksum != digest(data)
            or data.get("architecture_version") != "v2_contracts"
            or data.get("variant") != self.settings.variant
        ):
            raise ValueError("invalid GP checkpoint checksum/version")
        events = data["visible_events"]
        if len({e["event_id"] for e in events}) != len(events):
            raise ValueError("duplicate checkpoint visible events")
        interaction = InteractionIndex(
            completeness=data["interaction"]["completeness"],
            imported_event_count=data["interaction"]["imported_event_count"],
        )
        for event in events:
            if event.get("receipt"):
                validate_receipt(event["receipt"], event["content"], event["event_id"])
                interaction.apply(event["receipt"], event["turn"])
        if interaction.requests != data["interaction"]["requests"]:
            raise ValueError("checkpoint request facts do not match receipts")
        snapshot = SemanticSnapshot.model_validate(data["snapshot"]) if data["snapshot"] else None
        history = [ChatMessage(role=e["role"], content=e["content"]) for e in events]
        self._history, self._events = history, events
        self._episode_id, self._state = data["episode_id"], snapshot
        self._registry = IdentityIndex(entries=data["identities"])
        self._interactions, self._bridge = interaction, data["bridge"]
        self._version, self._turn_counter = data["version"], data["turn_counter"]
        self._metadata = {}

    def restore_events(self, records):
        """Restore a complete pipeline audit; only externally acknowledged replies count.

        Native write-ahead commit records alone cannot prove a reply was returned.
        Visible-only imports need their checkpoint, not an incomplete event suffix.
        This method performs no model calls.
        """
        started, committed, visible = {}, {}, []
        last, turn_counter = None, 0
        for record in records:
            kind, payload = record.get("event_type"), record.get("payload", {})
            if kind == "assistant_gp_turn_started":
                started[payload["turn_id"]] = payload
            elif kind == "assistant_gp_state_committed":
                committed[payload["turn_id"]] = payload
            elif kind == "assistant_generation_completed":
                receipt = payload.get("llm_call", {}).get("goal_progression", {}).get("receipt")
                if receipt is None:
                    raise ValueError("audit contains a reply without a GP v2 receipt")
                commit = next((p for p in committed.values() if p["receipt"] == receipt), None)
                if commit is None or commit["turn_id"] not in started:
                    raise ValueError("audit lacks complete native GP commit records")
                boundary = commit["history_boundary"]
                if boundary["imported_event_count"]:
                    raise ValueError(
                        "restore the imported-history checkpoint before replaying this suffix"
                    )
                if last is not None and commit["episode_id"] != last["episode_id"]:
                    raise ValueError("audit mixes GP episodes")
                pending = started[commit["turn_id"]]["user_event"]
                reply = payload["assistant_message"]
                if reply != commit["final_reply"] or pending["event_id"] != f"m{len(visible) + 1}":
                    raise ValueError("audit contains mismatched text or a missing visible prefix")
                turn_counter = int(commit["turn_id"].removeprefix("t"))
                blocks = {}
                for part in receipt["delivered_units"]:
                    blocks[part["unit_id"]] = {**part["text_span"], "text_hash": part["text_hash"]}
                for index, part in enumerate(receipt["issued_requests"]):
                    blocks[f"request:{index}:{part['need_id']}"] = {
                        **part["text_span"],
                        "text_hash": part["text_hash"],
                    }
                visible.extend(
                    [
                        pending,
                        {
                            "event_id": receipt["reply_event_id"],
                            "role": "assistant",
                            "content": reply,
                            "blocks": blocks,
                            "receipt": receipt,
                            "turn": turn_counter,
                        },
                    ]
                )
                last = commit
        if last is None:
            raise ValueError("audit contains no acknowledged GP v2 replies")
        checkpoint = {
            "architecture_version": last["architecture_version"],
            "variant": last["variant"],
            "episode_id": last["episode_id"],
            "visible_events": visible,
            "snapshot": last["state"],
            "version": last["snapshot_version"],
            "turn_counter": turn_counter,
            "identities": last["identity_index"],
            "interaction": {**last["history_boundary"], "requests": last["interaction_after"]},
            "bridge": last["execution_bridge"],
        }
        self.restore_checkpoint({**checkpoint, "checksum": digest(checkpoint)})

    async def _visible_tokens(self, text, runtime):
        result = {"visible_response_tokens": None, "visible_token_source": "unknown"}
        counter = getattr(self.client, "count_visible_tokens", None)
        if counter is None:
            return result
        while True:
            try:
                async with runtime.semaphore:
                    async with asyncio.timeout(self.settings.recovery.call_timeout_seconds):
                        counted = await counter(text)
                count = counted.get("visible_response_tokens")
                if count is not None and (type(count) is not int or count < 0):
                    raise ValueError("invalid visible token count")
                return counted
            except (ModelRequestError, TimeoutError, ValueError) as exc:
                error = ContractError(
                    "call.timeout" if isinstance(exc, TimeoutError) else "call.transient",
                    "runtime",
                    "Visible tokenizer unavailable",
                )
                try:
                    runtime.repair(error)
                except RecoveryExhausted:
                    runtime.recorder.emit("assistant_gp_tokenizer_unavailable", error.payload())
                    result["visible_token_source"] = "tokenizer_unavailable"
                    return result

    async def respond(self, user_message, *, audit_sink=None):
        async with self._lock:
            self._turn_counter += 1
            turn_id = f"t{self._turn_counter}"
            start = time.perf_counter()
            ledger = RetryLedger(self.settings)
            audit = {
                "architecture_version": "v2_contracts",
                "variant": self.settings.variant,
                "config_hash": digest(
                    {
                        "settings": self.settings.model_dump(mode="json"),
                        "generation": {
                            k: v.model_dump() for k, v in self.profile.generation.items()
                        },
                        "prompts": {k: v.hash for k, v in self.prompts.items()},
                        "model_id": self.profile.model_id,
                        "base_url": self.profile.base_url,
                    }
                ),
                "calls": [],
                "degradations": [],
                "committed": False,
                "previous_state": self._state.model_dump() if self._state else None,
            }
            self._metadata = {
                "goal_progression": audit,
                "visible_response_tokens": 0,
                "visible_token_source": "not_delivered",
            }
            recorder = AuditRecorder(
                audit_sink, ledger, self._episode_id, turn_id, self.settings.variant, audit
            )
            pending = {
                "event_id": f"m{len(self._events) + 1}",
                "role": "user",
                "content": user_message,
            }
            store = EvidenceStore([*self._events, pending])
            runtime = ContractRuntime(
                self.client, self.profile, self.prompts, recorder, ledger, audit
            )
            engine = TurnEngine(
                runtime,
                store,
                self._registry,
                self._interactions,
                self._state,
                self._bridge,
                turn_id,
                self._version,
            )
            try:
                recorder.emit(
                    "assistant_gp_turn_started",
                    {
                        "user_event": pending,
                        "snapshot_version": self._version,
                        "owner_map": self._owner_map(),
                    },
                )
                async with asyncio.timeout(self.settings.turn_timeout_seconds):
                    content, candidate, registry, receipt, blocks = await engine.execute(
                        f"m{len(self._events) + 2}"
                    )
                    token_metadata = await self._visible_tokens(content, runtime)
                next_interactions = copy.deepcopy(self._interactions)
                next_interactions.apply(receipt, self._turn_counter)
                next_registry = registry.committed(self._registry, candidate)
                assistant_event = {
                    "event_id": receipt["reply_event_id"],
                    "role": "assistant",
                    "content": content,
                    "blocks": blocks,
                    "receipt": receipt,
                    "turn": self._turn_counter,
                }
                next_state = candidate if self.settings.variant != "no_tracker" else None
                next_history = [
                    *self._history,
                    ChatMessage(role="user", content=user_message),
                    ChatMessage(role="assistant", content=content),
                ]
                next_events = [*self._events, pending, assistant_event]
                next_bridge = {
                    "contract_id": receipt["contract_id"],
                    "delivered_unit_ids": [r["unit_id"] for r in receipt["delivered_units"]],
                    "issued_need_ids": [r["need_id"] for r in receipt["issued_requests"]],
                    "reply_event_id": receipt["reply_event_id"],
                }
                committed_state = next_state.model_dump() if next_state else None
                # Prepare every potentially-failing audit write before the no-await state swap.
                recorder.emit(
                    "assistant_gp_commit_prepared",
                    {
                        "receipt": receipt,
                        "candidate_snapshot": candidate.model_dump(),
                        "interaction_before": self._interactions.requests,
                        "interaction_after": next_interactions.requests,
                    },
                )
                recorder.emit(
                    "assistant_gp_state_committed",
                    {
                        "committed": True,
                        "final_reply": content,
                        "receipt": receipt,
                        "state": committed_state,
                        "snapshot_version": engine.version,
                        "interaction_after": next_interactions.requests,
                        "identity_index": next_registry.entries,
                        "execution_bridge": next_bridge,
                        "history_boundary": {
                            "completeness": next_interactions.completeness,
                            "imported_event_count": next_interactions.imported_event_count,
                        },
                        **token_metadata,
                    },
                )
                # A single synchronous, non-failing assignment boundary. No later sink call.
                self._history, self._events = next_history, next_events
                self._state, self._registry = next_state, next_registry
                self._interactions, self._version = next_interactions, engine.version
                self._bridge = next_bridge
                audit.update(
                    committed=True,
                    final_reply=content,
                    receipt=receipt,
                    committed_state=committed_state,
                )
                self._metadata.update(token_metadata)
                return content
            except BaseException as exc:
                reason = (
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "turn_deadline"
                    if isinstance(exc, TimeoutError)
                    else "recovery_exhausted"
                    if isinstance(exc, RecoveryExhausted)
                    else "fatal"
                )
                audit.update(
                    error_type=type(exc).__name__,
                    termination_reason=reason,
                    retry_counts=ledger.snapshot(),
                )
                try:
                    recorder.emit(
                        "assistant_gp_turn_failed",
                        {
                            "committed": False,
                            "error_type": type(exc).__name__,
                            "retry_counts": ledger.snapshot(),
                            "visible_response_tokens": 0,
                            "snapshot_version": self._version,
                            "termination_reason": reason,
                        },
                    )
                except RecoveryExhausted:
                    # A persistently failed sink cannot log its own failure. Preserve the cause.
                    audit["failure_event_persisted"] = False
                raise
            finally:
                audit["wall_seconds"] = time.perf_counter() - start
                audit["retry_counts"] = ledger.snapshot()
                self._metadata["internal_call_count"] = len(audit["calls"])
                for name in ("input_tokens", "output_tokens"):
                    known = [c.get("usage", {}).get(name) for c in audit["calls"]]
                    known = [n for n in known if type(n) is int and n >= 0]
                    self._metadata[f"internal_{name}"] = sum(known)
                self._metadata["internal_usage_complete"] = all(
                    type(c.get("usage", {}).get(name)) is int
                    for c in audit["calls"]
                    for name in ("input_tokens", "output_tokens")
                )

    def _owner_map(self):
        semantic = "tracker" if self.settings.variant != "no_tracker" else "partitioned_planners"
        current = (
            "generator"
            if self.settings.variant == "no_intra"
            else "joint"
            if self.settings.variant == "joint"
            else "intra"
        )
        adjacent = (
            None
            if self.settings.variant == "no_inter"
            else "joint"
            if self.settings.variant == "joint"
            else "inter"
        )
        return {
            "semantic": semantic,
            "current_policy": current,
            "adjacent_policy": adjacent,
            "authorization": "assembler",
            "expression": "generator",
            "facts": "runtime",
        }
