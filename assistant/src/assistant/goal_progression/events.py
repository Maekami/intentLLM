"""Visible evidence, stable identity metadata and receipt-derived interaction facts."""

import copy
import hashlib
import json
from dataclasses import dataclass, field

from .errors import ContractError
from .identity import validate_reference
from .schemas import EvidenceRef, SemanticSnapshot


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class EvidenceStore:
    def __init__(self, events: list[dict]):
        self.events = copy.deepcopy(events)
        self.by_id = {e["event_id"]: e for e in self.events}
        self.positions = {e["event_id"]: i for i, e in enumerate(self.events)}

    def resolve(self, ref: EvidenceRef, owner="tracker", path=None) -> dict:
        event = self.by_id.get(ref.event_id)
        if event is None:
            raise ContractError(
                "contract.reference",
                owner,
                f"Unknown visible event {ref.event_id!r}. Available event IDs: {list(self.by_id)}.",
                path or f"evidence:{ref.event_id}",
            )
        text = event["content"]
        if ref.block_id is not None:
            block = event.get("blocks", {}).get(ref.block_id)
            if block is None:
                raise ContractError(
                    "contract.reference",
                    owner,
                    f"Block {ref.block_id!r} does not belong to event {ref.event_id!r}. "
                    f"Allowed block IDs for that event: {[None, *event.get('blocks', {})]}. "
                    "Choose the correct original event/block pair; null refers to the whole event.",
                    path or f"evidence:{ref.event_id}:{ref.block_id}",
                )
            text = text[block["start"] : block["end"]]
        return {
            "event_id": ref.event_id,
            "block_id": ref.block_id,
            "role": event["role"],
            "content": text,
        }

    def project(self, refs: list[EvidenceRef], owner="runtime") -> list[dict]:
        seen = set()
        result = []
        # A whole message already contains each of its blocks.
        whole = {r.event_id for r in refs if r.block_id is None}
        for ref in refs:
            key = (ref.event_id, ref.block_id)
            if key in seen or (ref.block_id is not None and ref.event_id in whole):
                continue
            seen.add(key)
            result.append(self.resolve(ref, owner))
        return result

    def is_user(self, ref: EvidenceRef) -> bool:
        return self.resolve(ref)["role"] == "user"


@dataclass
class IdentityIndex:
    entries: dict[str, dict] = field(default_factory=dict)
    # Only current transaction aliases; not historical semantic state.
    aliases: dict[str, str] = field(default_factory=dict)

    def clone(self):
        return copy.deepcopy(self)

    def allocate(self, ref, kind, origin, turn_id, owner) -> str:
        validate_reference(ref, kind, self, owner, ref)
        if ref in self.entries:
            return ref
        key = f"{turn_id}:{owner}:{kind}:{ref}"
        if key in self.aliases:
            return self.aliases[key]
        prefix = {"goal": "g", "constraint": "c", "need": "n"}[kind]
        index = 1
        while f"{prefix}{index:04d}" in self.entries:
            index += 1
        canonical = f"{prefix}{index:04d}"
        self.entries[canonical] = {
            "kind": kind,
            "origin_event_id": origin,
            "created_turn_id": turn_id,
            "scope_id": None,
        }
        self.aliases[key] = canonical
        return canonical

    def annotated_history(self, store: EvidenceStore) -> list[dict]:
        annotations: dict[str, list[dict]] = {}
        for ref, entry in self.entries.items():
            annotations.setdefault(entry["origin_event_id"], []).append(
                {
                    "id": ref,
                    "kind": entry["kind"],
                    "origin_goal_id": entry["scope_id"],
                }
            )
        result = []
        for event in store.events:
            item = {k: copy.deepcopy(event[k]) for k in ("event_id", "role", "content")}
            if event.get("blocks"):
                item["blocks"] = copy.deepcopy(event["blocks"])
            if annotations.get(event["event_id"]):
                item["identity_refs"] = annotations[event["event_id"]]
            result.append(item)
        return result

    def committed(self, previous: "IdentityIndex", snapshot: SemanticSnapshot):
        keep = set(previous.entries) | {
            item.ref
            for group in (snapshot.goals, snapshot.constraints, snapshot.needs)
            for item in group
        }
        return IdentityIndex(
            entries={
                key: copy.deepcopy(value) for key, value in self.entries.items() if key in keep
            }
        )


@dataclass
class InteractionIndex:
    requests: dict[str, dict] = field(default_factory=dict)
    receipt_ids: set[str] = field(default_factory=set)
    completeness: str = "complete"
    imported_event_count: int = 0

    def view(self, snapshot: SemanticSnapshot, registry: IdentityIndex, store: EvidenceStore):
        result = {}
        for need in snapshot.needs:
            fact = copy.deepcopy(self.requests.get(need.ref))
            origin = registry.entries[need.ref]["origin_event_id"]
            newly_known = (
                self.completeness == "complete"
                or store.positions.get(origin, -1) >= self.imported_event_count
            )
            known = fact is not None or newly_known
            last_position = store.positions.get((fact or {}).get("last_issued_event_id", ""), -1)
            auth = need.renewed_authorization_ref
            auth_position = store.positions.get(auth.event_id, -1) if auth else -1
            boundary = self.imported_event_count - 1 if not known else -1
            renewed = auth is not None and auth_position > max(last_position, boundary)
            eligible = need.status == "unknown" and ((known and fact is None) or renewed)
            result[need.ref] = {
                "issued_count": (fact or {}).get("issued_count", 0) if known else None,
                "last_issued_event_id": (fact or {}).get("last_issued_event_id"),
                "last_issued_turn": (fact or {}).get("last_issued_turn"),
                "history_known": known,
                "request_eligible": eligible,
                "reason": (
                    "not_unknown"
                    if need.status != "unknown"
                    else "renewed_authorization"
                    if renewed
                    else "first_issue"
                    if eligible
                    else "already_issued"
                    if known
                    else "history_unknown"
                ),
            }
        return result

    def apply(self, receipt: dict, turn: int):
        if receipt["receipt_id"] in self.receipt_ids:
            return
        for request in receipt["issued_requests"]:
            old = self.requests.get(request["need_id"], {})
            self.requests[request["need_id"]] = {
                "issued_count": old.get("issued_count", 0) + 1,
                "last_issued_event_id": receipt["reply_event_id"],
                "last_issued_turn": turn,
            }
        self.receipt_ids.add(receipt["receipt_id"])


class AuditRecorder:
    """Synchronous native audit hook; at-least-once with stable event IDs on I/O retry."""

    def __init__(self, sink, ledger, episode_id, turn_id, variant, audit):
        self.sink, self.ledger = sink, ledger
        self.episode_id, self.turn_id, self.variant = episode_id, turn_id, variant
        self.audit = audit
        self.sequence = 0

    def emit(self, kind: str, payload: dict):
        self.sequence += 1
        item = copy.deepcopy(
            {
                **payload,
                "architecture_version": "v2_contracts",
                "schema_version": 2,
                "config_hash": self.audit.get("config_hash"),
                "episode_id": self.episode_id,
                "turn_id": self.turn_id,
                "variant": self.variant,
                "event_id": f"{self.turn_id}:a{self.sequence}",
            }
        )
        if self.sink is None:
            return
        while True:
            try:
                self.sink(kind, copy.deepcopy(item))
                return
            except OSError:
                error = ContractError("runtime.io", "runtime", "Native audit sink I/O failed")
                self.ledger.consume([error])
                # Never recurse into the failing sink to report its own error.
                self.audit.setdefault("audit_io_retries", []).append(
                    {
                        "event_id": item["event_id"],
                        "retry_counts": self.ledger.snapshot(),
                    }
                )
