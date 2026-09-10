"""Compile candidates into one execution contract. Never reinterpret free text."""

from .events import digest
from .schemas import BlockedCurrent, DeliveryUnit, SelectedRequest, TurnContract


def assemble(snapshot, intra, inter, policy, *, version, turn_id):
    deliveries, requests, blocked, deferred = [], [], [], []
    if intra is not None:
        for item in intra.items:
            for index, proposal in enumerate(item.deliveries):
                deliveries.append(
                    DeliveryUnit(
                        **proposal.model_dump(),
                        unit_id=f"intra:{item.goal_id}:{index}",
                        source="intra",
                        criticality="required",
                    )
                )
            if item.request:
                requests.append(
                    SelectedRequest(
                        **item.request.model_dump(), source="intra", criticality="required"
                    )
                )
            if item.blocked_by:
                blocked.append(BlockedCurrent(goal_id=item.goal_id, need_ids=item.blocked_by))
    if inter.request:
        requests.append(
            SelectedRequest(**inter.request.model_dump(), source="inter", criticality="optional")
        )
    selected, seen = [], set()
    for request in requests:
        if request.need_id not in seen and len(selected) < policy.request_budget:
            seen.add(request.need_id)
            selected.append(request)
        else:
            deferred.append(
                {
                    "source": request.source,
                    "need_id": request.need_id,
                    "reason": "request_budget_or_duplicate",
                }
            )
    selected_ids = {r.need_id for r in selected}
    if intra is not None:
        for item in intra.items:
            waiting = list(item.blocked_by)
            if item.request and item.request.need_id not in selected_ids:
                waiting.append(item.request.need_id)
            if (
                not item.deliveries
                and waiting
                and not (item.request and item.request.need_id in selected_ids)
            ):
                deliveries.append(
                    DeliveryUnit(
                        unit_id=f"boundary:{item.goal_id}",
                        source="runtime",
                        kind="boundary",
                        criticality="required",
                        goal_id=item.goal_id,
                        target=(
                            "Briefly state the current input limitation without another request, "
                            "invitation or claim of completion. Do not fabricate missing content."
                        ),
                        required_need_ids=[],
                        material_refs=[],
                    )
                )
    for index, proposal in enumerate(inter.deliveries):
        deliveries.append(
            DeliveryUnit(
                **proposal.model_dump(),
                unit_id=f"inter:{inter.goal_id}:{index}",
                source="inter",
                criticality="optional",
            )
        )
    payload = {
        "snapshot_version": version,
        "deliveries": [d.model_dump() for d in deliveries],
        "requests": [r.model_dump() for r in selected],
        "blocked_current": [b.model_dump() for b in blocked],
    }
    contract = TurnContract(contract_id=f"{turn_id}:{digest(payload)[:16]}", **payload)
    return contract, deferred


def without_optional(contract, *, unit_ids=None):
    result = contract.model_copy(deep=True)
    result.deliveries = [
        unit
        for unit in result.deliveries
        if unit.criticality == "required" or (unit_ids is not None and unit.unit_id not in unit_ids)
    ]
    if unit_ids is None:
        result.requests = [r for r in result.requests if r.criticality == "required"]
    result.contract_id = f"{contract.contract_id}:pruned:{digest(result.model_dump())[:12]}"
    return result
