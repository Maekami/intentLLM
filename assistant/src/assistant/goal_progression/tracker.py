"""Validate a single semantic owner, then assign runtime-owned canonical IDs."""

import copy

from .anchors import is_progress_anchor, progress_anchor_ids
from .errors import ContractError
from .events import EvidenceStore, IdentityIndex
from .identity import validate_semantic_identities
from .schemas import SemanticSnapshot


def normalize_projection(raw, registry, store, turn_id, policy):
    """Adapt layout only; never choose or silently rewrite a model's goal scope."""
    for scope in ("current", "adjacent"):
        for index, goal in enumerate(getattr(raw, scope + "_goals")):
            if goal.scope != scope:
                raise ContractError(
                    "contract.reference",
                    "tracker",
                    f"{scope}_goals may contain only {scope} goals; keep scope and list consistent",
                    f"$.{scope}_goals[{index}].scope",
                )
    snapshot = SemanticSnapshot(
        goals=[*raw.current_goals, *raw.adjacent_goals],
        constraints=raw.constraints,
        needs=raw.needs,
        changed_goal_ids=raw.changed_goal_ids,
    )
    try:
        return normalize_snapshot(snapshot, registry, store, turn_id, policy)
    except ContractError as error:
        prefix = "$.goals["
        if error.field_path.startswith(prefix):
            index_text, end, tail = error.field_path[len(prefix) :].partition("]")
            if end and index_text.isdigit():
                index = int(index_text)
                scope = "current" if index < len(raw.current_goals) else "adjacent"
                offset = index if scope == "current" else index - len(raw.current_goals)
                error.field_path = f"$.{scope}_goals[{offset}]{tail}"
        elif error.field_path == "goals":
            error.field_path = "$"
        raise


def normalize_snapshot(
    raw: SemanticSnapshot,
    registry: IdentityIndex,
    store: EvidenceStore,
    turn_id: str,
    policy,
    *,
    owner="tracker",
    owners=None,
    require_current=True,
):
    validate_semantic_identities(raw, registry, owner, owners=owners)
    result = raw.model_copy(deep=True)
    candidate = registry.clone()
    mapping = {}
    declared_kinds = {}
    owners = owners or {}
    groups = (("goal", result.goals), ("constraint", result.constraints), ("need", result.needs))
    for kind, items in groups:
        group = {"goal": "goals", "constraint": "constraints", "need": "needs"}[kind]
        for index, item in enumerate(items):
            who = owners.get(item.ref, owner)
            if item.ref in mapping:
                raise ContractError(
                    "contract.reference",
                    who,
                    f"Duplicate semantic ID {item.ref!r}; already declared as a "
                    f"{declared_kinds[item.ref]} entity in this output. Each declaration "
                    "needs a globally unique ref across goals, constraints and needs. "
                    "Declare the same entity only once. For distinct new entities, choose "
                    "distinct new:<local_name> refs and update their linked references. "
                    "Preserve registered identities; do not silently merge different meanings.",
                    f"$.{group}[{index}].ref",
                )
            if not item.evidence_refs:
                raise ContractError(
                    "contract.reference",
                    who,
                    "Origin evidence is required",
                    f"$.{group}[{index}].evidence_refs",
                )
            for offset, ref in enumerate(item.evidence_refs):
                store.resolve(ref, who, f"$.{group}[{index}].evidence_refs[{offset}]")
            origin = min(item.evidence_refs, key=lambda r: store.positions[r.event_id]).event_id
            mapping[item.ref] = candidate.allocate(item.ref, kind, origin, turn_id, who)
            declared_kinds[item.ref] = kind

    def mapped(ref, who, path=None):
        if ref not in mapping:
            raise ContractError(
                "contract.reference",
                who,
                f"Reference {ref!r} is not declared in this output. Declare the referenced "
                "entity in this snapshot or remove/correct the dangling link. "
                f"Declared refs: {list(mapping)}. Do not invent canonical IDs.",
                path or ref,
            )
        return mapping[ref]

    source_owners = {}
    user_events = [e["event_id"] for e in store.events if e["role"] == "user"]
    for index, goal in enumerate(result.goals):
        who = owners.get(goal.ref, owner)
        source_owners[mapping[goal.ref]] = who
        goal.ref = mapping[goal.ref]
        goal.anchor_goal_ids = [
            mapped(ref, who, f"$.goals[{index}].anchor_goal_ids[{offset}]")
            for offset, ref in enumerate(goal.anchor_goal_ids)
        ]
        goal.constraint_ids = []  # Derived below from the authoritative applies_to edges.
        for offset, ref in enumerate(goal.material_refs):
            store.resolve(ref, who, f"$.goals[{index}].material_refs[{offset}]")
        if not goal.description.strip() or (
            goal.status == "active" and not goal.remaining_work.strip()
        ):
            raise ContractError(
                "contract.coverage",
                who,
                "Active goals need a description and remaining work",
                f"$.goals[{index}].remaining_work",
            )
        if goal.basis == "explicit" and not any(store.is_user(r) for r in goal.evidence_refs):
            raise ContractError(
                "contract.reference",
                who,
                "An explicit goal must cite the user event that supports the request, not "
                "only an assistant reply. If it is only inferred, label its basis inferred. "
                f"Available user event IDs: {user_events}.",
                f"$.goals[{index}].evidence_refs",
            )
    goals = {g.ref: g for g in result.goals}
    current = {g.ref for g in result.goals_for("current")}
    anchors = progress_anchor_ids(result)
    if require_current and not current:
        raise ContractError(
            "contract.coverage",
            owner,
            "Represent the latest user's response obligation as an active current goal",
            "goals",
        )
    if len(result.goals_for("adjacent")) > policy.adjacent_candidate_limit:
        who = source_owners[result.goals_for("adjacent")[0].ref]
        raise ContractError("contract.coverage", who, "Adjacent candidate limit exceeded", "goals")
    for index, goal in enumerate(result.goals):
        who = source_owners[goal.ref]
        if (
            goal.scope == "adjacent"
            and goal.status == "active"
            and (not goal.anchor_goal_ids or not set(goal.anchor_goal_ids) <= anchors)
        ):
            raise ContractError(
                "contract.reference",
                who,
                "Active adjacent goals need an active current goal or an addressed result "
                "as a progress anchor, not a paused goal or an unfulfilled adjacent candidate. "
                f"Allowed refs in this projection: {[ref for ref, target in mapping.items() if target in anchors]}.",
                f"$.goals[{index}].anchor_goal_ids",
            )
        if goal.scope == "current" and goal.anchor_goal_ids:
            raise ContractError(
                "contract.reference",
                who,
                "Current goals do not need adjacent anchors",
                f"$.goals[{index}].anchor_goal_ids",
            )

    for index, constraint in enumerate(result.constraints):
        who = owners.get(constraint.ref, owner)
        constraint.ref = mapping[constraint.ref]
        constraint.applies_to = [
            mapped(ref, who, f"$.constraints[{index}].applies_to[{offset}]")
            for offset, ref in enumerate(constraint.applies_to)
        ]
        if not constraint.applies_to or not set(constraint.applies_to) <= goals.keys():
            raise ContractError(
                "contract.reference",
                who,
                "applies_to must be a NONEMPTY list of goal refs declared in this output: "
                f"{[g.ref for g in raw.goals]}. If the constraint is no longer relevant, "
                "omit it and its links; do not retain an orphan constraint. Preserve its "
                "actual applicability and conditions rather than attaching it arbitrarily.",
                f"$.constraints[{index}].applies_to",
            )
        if constraint.basis == "explicit" and not any(
            store.is_user(r) for r in constraint.evidence_refs
        ):
            raise ContractError(
                "contract.reference",
                who,
                "An explicit constraint needs the supporting user event; assistant examples "
                f"are not user preferences. Available user event IDs: {user_events}.",
                f"$.constraints[{index}].evidence_refs",
            )
    for goal in result.goals:
        # One authoritative relation, one deterministic inverse. Never reconcile
        # two separately model-authored guesses about constraint applicability.
        goal.constraint_ids = sorted(c.ref for c in result.constraints if goal.ref in c.applies_to)

    for index, need in enumerate(result.needs):
        who = owners.get(need.ref, owner)
        need.ref, need.goal_id = (
            mapping[need.ref],
            mapped(need.goal_id, who, f"$.needs[{index}].goal_id"),
        )
        if need.goal_id not in goals:
            raise ContractError(
                "contract.reference", who, "Need scope is not a goal", f"$.needs[{index}].goal_id"
            )
        # Stable provenance is not a foreign-key requirement for the current
        # projection. Tracker owns relevance; runtime never semantically rebinds.
        # Request history is keyed only by need.ref, so reassociation cannot reset it.
        if candidate.entries[need.ref]["scope_id"] is None:
            candidate.entries[need.ref]["scope_id"] = need.goal_id
        for offset, ref in enumerate(need.answer_refs):
            store.resolve(ref, who, f"$.needs[{index}].answer_refs[{offset}]")
        if need.status == "available" and not need.answer_refs:
            raise ContractError(
                "contract.dependency",
                who,
                "Available information needs answer references",
                f"$.needs[{index}].answer_refs",
            )
        if need.status == "unavailable" and not any(store.is_user(r) for r in need.evidence_refs):
            raise ContractError(
                "contract.reference",
                who,
                "Unavailable needs require user evidence",
                f"$.needs[{index}].evidence_refs",
            )
        if need.renewed_authorization_ref:
            store.resolve(
                need.renewed_authorization_ref, who, f"$.needs[{index}].renewed_authorization_ref"
            )
            if not store.is_user(need.renewed_authorization_ref):
                raise ContractError(
                    "contract.reference",
                    who,
                    "Renewed authorization must cite a user event",
                    f"$.needs[{index}].renewed_authorization_ref",
                )
    result.changed_goal_ids = [
        mapped(ref, owner, f"$.changed_goal_ids[{index}]")
        for index, ref in enumerate(result.changed_goal_ids)
    ]
    if not set(result.changed_goal_ids) <= goals.keys():
        raise ContractError(
            "contract.reference", owner, "Changed IDs must be goals", "$.changed_goal_ids"
        )
    return result, candidate, mapping


def remap_plan(plan, mapping):
    result = plan.model_copy(deep=True)
    items = result.items if hasattr(result, "items") else [result]
    for item in items:
        if item.goal_id is not None:
            item.goal_id = mapping.get(item.goal_id, item.goal_id)
        for delivery in item.deliveries:
            delivery.goal_id = mapping.get(delivery.goal_id, delivery.goal_id)
            delivery.required_need_ids = [
                mapping.get(ref, ref) for ref in delivery.required_need_ids
            ]
        if item.request:
            item.request.need_id = mapping.get(item.request.need_id, item.request.need_id)
        if hasattr(item, "blocked_by"):
            item.blocked_by = [mapping.get(ref, ref) for ref in item.blocked_by]
    return result


def namespace_packet(packet, role):
    packet = packet.model_copy(deep=True)
    mapping = {
        item.ref: f"new:{role}/{item.ref[4:]}"
        for group in (packet.semantic.goals, packet.semantic.constraints, packet.semantic.needs)
        for item in group
        if item.ref.startswith("new:")
    }
    data = packet.semantic.model_dump()
    for group in ("goals", "constraints", "needs"):
        for item in data[group]:
            item["ref"] = mapping.get(item["ref"], item["ref"])
            for field in ("anchor_goal_ids", "constraint_ids", "applies_to"):
                if field in item:
                    item[field] = [mapping.get(x, x) for x in item[field]]
            if "goal_id" in item:
                item["goal_id"] = mapping.get(item["goal_id"], item["goal_id"])
    data["changed_goal_ids"] = [mapping.get(x, x) for x in data["changed_goal_ids"]]
    packet.semantic = SemanticSnapshot.model_validate(data)
    packet.plan = remap_plan(packet.plan, mapping)
    return packet


def merge_packets(intra, inter, registry, store, turn_id, policy):
    intra, inter = namespace_packet(intra, "intra"), namespace_packet(inter, "inter")
    for packet, scope, owner in ((intra, "current", "intra"), (inter, "adjacent", "inter")):
        if any(goal.scope != scope for goal in packet.semantic.goals):
            raise ContractError(
                "contract.reference",
                owner,
                "Packet contains another role's goals",
                "semantic.goals",
            )
    current = intra.semantic.goals
    for goal in inter.semantic.goals:
        resolved = []
        for anchor in goal.anchor_goal_ids:
            if anchor in store.by_id:
                resolved.extend(
                    g.ref
                    for g in current
                    if is_progress_anchor(g) and any(r.event_id == anchor for r in g.evidence_refs)
                )
            else:
                resolved.append(anchor)
        goal.anchor_goal_ids = list(dict.fromkeys(resolved))
    combined = SemanticSnapshot(goals=[], constraints=[], needs=[], changed_goal_ids=[])
    owners = {}
    for packet, owner in ((intra, "intra"), (inter, "inter")):
        for group in ("goals", "constraints", "needs"):
            for item in getattr(packet.semantic, group):
                if item.ref in owners:
                    raise ContractError(
                        "contract.reference",
                        owner,
                        "Packets cannot duplicate a canonical entity",
                        item.ref,
                    )
                owners[item.ref] = owner
                getattr(combined, group).append(copy.deepcopy(item))
        combined.changed_goal_ids.extend(packet.semantic.changed_goal_ids)
    snapshot, candidate, mapping = normalize_snapshot(
        combined, registry, store, turn_id, policy, owner="intra", owners=owners
    )
    return snapshot, candidate, remap_plan(intra.plan, mapping), remap_plan(inter.plan, mapping)
