"""Explicit task-scoped contexts. Required original materials are never truncated."""

import copy

from .anchors import is_progress_anchor
from .events import digest
from .identity import registered_ids
from .schemas import EvidenceRef


class ContextBuilder:
    def __init__(self, store, registry, interactions):
        self.store, self.registry, self.interactions = store, registry, interactions

    @staticmethod
    def evidence_domain(context):
        """Only event/block pairs whose original text is visible to this role."""
        pairs = {}

        def add(event_id, block_id=None):
            pairs.setdefault(event_id, set()).add(block_id)

        for event in context.get("visible_history", []):
            add(event["event_id"])
            for block_id in event.get("blocks", {}):
                add(event["event_id"], block_id)
        if context.get("latest_user"):
            add(context["latest_user"]["event_id"])
        for key in ("evidence", "inter_materials"):
            for event in context.get(key, []):
                add(event["event_id"], event.get("block_id"))
        return {
            key: ([None] if None in blocks else []) + sorted(b for b in blocks if b is not None)
            for key, blocks in pairs.items()
        }

    @property
    def latest(self):
        event = self.store.events[-1]
        return {k: event[k] for k in ("event_id", "role", "content")}

    def without_latest_copy(self, evidence):
        visible = {self.latest["event_id"]}
        return [event for event in evidence if event["event_id"] not in visible]

    def tracker(self, previous, bridge):
        previous_wire = previous.model_dump() if previous else None
        if previous_wire:
            previous_wire.pop("changed_goal_ids", None)
            for goal in previous_wire["goals"]:
                goal.pop("constraint_ids", None)
        return {
            "visible_history": self.registry.annotated_history(self.store),
            "registered_ids": registered_ids(self.registry),
            "previous_snapshot": previous_wire,
            "interaction_view": (
                self.interactions.view(previous, self.registry, self.store) if previous else {}
            ),
            "previous_execution_bridge": copy.deepcopy(bridge),
            "history_completeness": self.interactions.completeness,
            "latest_user": self.latest,
        }

    def semantic_view(self, snapshot, goal_ids, *, include_evidence=True, need_ids=()):
        goals = [g for g in snapshot.goals if g.ref in goal_ids]
        constraints = [c for c in snapshot.constraints if set(c.applies_to) & set(goal_ids)]
        needs = [n for n in snapshot.needs if n.goal_id in goal_ids or n.ref in need_ids]
        refs = []
        for goal in goals:
            refs.extend(goal.material_refs)
            if include_evidence:
                refs.extend(goal.evidence_refs)
        for constraint in constraints:
            refs.extend(constraint.evidence_refs)
        for need in needs:
            refs.extend(need.answer_refs)
            refs.extend(need.evidence_refs)
        evidence = self.store.project(refs)
        # Tracker selects original material, including relevant prior replies.
        # Do not broadcast a whole reply merely because it is the most recent.
        evidence = self.without_latest_copy(evidence)
        return {
            "latest_user": self.latest,
            "goals": [g.model_dump() for g in goals],
            "constraints": [c.model_dump() for c in constraints],
            "needs": [n.model_dump() for n in needs],
            "evidence": evidence,
        }

    def planner(self, snapshot, role):
        scope = "current" if role == "intra" else "adjacent"
        ids = {g.ref for g in snapshot.goals_for(scope)}
        value = self.semantic_view(snapshot, ids)
        view = self.interactions.view(snapshot, self.registry, self.store)
        value["request_eligibility"] = {
            need.ref: view[need.ref] for need in snapshot.needs if need.goal_id in ids
        }
        value["changed_goal_ids"] = [ref for ref in snapshot.changed_goal_ids if ref in ids]
        if role == "inter":
            value["progress_anchors"] = self.progress_anchors(snapshot)
        return value

    @staticmethod
    def progress_anchors(snapshot):
        referenced = {
            ref for goal in snapshot.goals_for("adjacent") for ref in goal.anchor_goal_ids
        }
        return [
            {
                key: getattr(goal, key)
                for key in ("ref", "scope", "status", "description", "remaining_work")
            }
            for goal in snapshot.goals
            if is_progress_anchor(goal) and (goal.ref in referenced or goal.status == "active")
        ]

    def joint(self, snapshot):
        ids = {g.ref for g in snapshot.goals if g.status == "active"}
        value = self.semantic_view(snapshot, ids)
        value["request_eligibility"] = self.interactions.view(snapshot, self.registry, self.store)
        value["current_goal_ids"] = [g.ref for g in snapshot.goals_for("current")]
        value["adjacent_goal_ids"] = [g.ref for g in snapshot.goals_for("adjacent")]
        value["progress_anchors"] = self.progress_anchors(snapshot)
        return value

    def packet(self, role):
        return {
            "visible_history": self.registry.annotated_history(self.store),
            "registered_ids": registered_ids(self.registry),
            "owned_scope": "current" if role == "intra" else "adjacent",
            "request_facts": copy.deepcopy(self.interactions.requests),
            "history_completeness": self.interactions.completeness,
            "imported_event_count": self.interactions.imported_event_count,
            "latest_user": self.latest,
        }

    def generator(self, snapshot, contract, units=None, frozen=None):
        selected = contract.deliveries if units is None else units
        ids = {u.goal_id for u in selected}
        need_map = {n.ref: n for n in snapshot.needs}
        ids.update(need_map[r.need_id].goal_id for r in contract.requests)
        dependencies = {ref for unit in selected for ref in unit.required_need_ids}
        value = self.semantic_view(snapshot, ids, need_ids=dependencies)
        # Only the sealed decision is visible; no rejected proposals or request ledger.
        value["turn_contract"] = {
            "contract_id": contract.contract_id,
            "deliveries": [u.model_dump() for u in selected],
            "requests": [r.model_dump() for r in contract.requests],
        }
        extra = [ref for unit in selected for ref in unit.material_refs]
        refs = [
            EvidenceRef(event_id=e["event_id"], block_id=e["block_id"]) for e in value["evidence"]
        ]
        value["evidence"] = self.without_latest_copy(self.store.project([*refs, *extra]))
        if frozen:
            value["frozen_body_for_continuity"] = copy.deepcopy(frozen)
        return value

    def unit_signature(self, snapshot, unit):
        # Independent of unrelated Inter choices or global version numbers.
        value = self.semantic_view(snapshot, {unit.goal_id}, need_ids=unit.required_need_ids)
        value["unit"] = unit.model_dump()
        value["unit_materials"] = self.store.project(unit.material_refs)
        return digest(value)
