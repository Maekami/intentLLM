"""Local realization coverage checks; no second planning/review call."""

from .errors import ContractError


def validate_units(result, contract, requested_ids):
    allowed = {u.unit_id for u in contract.deliveries}
    ids = [u.unit_id for u in result.units]
    issues = [i.unit_id for i in result.issues]
    duplicates = {ref for ref in ids if ids.count(ref) > 1}
    duplicates |= {ref for ref in issues if issues.count(ref) > 1}
    if duplicates:
        raise ContractError(
            "contract.coverage",
            "generator",
            f"Duplicate output unit IDs: {sorted(duplicates)}. Put the complete body for each "
            "unit in ONE text field; do not split its paragraphs into repeated unit entries.",
            "$.units / $.issues",
            affected_units=sorted(duplicates),
        )
    unknown = (set(ids) | set(issues)) - allowed
    if unknown:
        raise ContractError(
            "contract.reference",
            "generator",
            f"Unapproved unit IDs: {sorted(unknown)}. Only realize the exact supplied IDs: "
            f"{sorted(requested_ids)}. Do not create new units.",
            "$.units / $.issues",
            affected_units=sorted(unknown),
        )
    overlap = set(ids) & set(issues)
    if overlap:
        raise ContractError(
            "contract.reference",
            "generator",
            f"Units appear in BOTH units and issues: {sorted(overlap)}. For each unit, "
            "choose exactly one: a COMPLETE body in units with NO issue; OR an issue "
            "with NO body. A preface or promise is not a completed body.",
            "$.units / $.issues",
            affected_units=sorted(overlap),
        )
    # A repair may return the full output; accepted frozen bodies are not overwritten.
    missing = set(requested_ids) - set(ids) - set(issues)
    blank = {u.unit_id for u in result.units if not u.text.strip()}
    if missing or blank:
        raise ContractError(
            "contract.coverage",
            "generator",
            f"Missing units: {sorted(missing)}; blank bodies: {sorted(blank)}. Return a "
            "complete body or a local issue for each requested ID, exactly once.",
            "$.units / $.issues",
            affected_units=sorted(missing | blank),
        )
