"""The two action owners share structural validation, never each other's proposals."""

from .contracts import validate_inter, validate_intra


def check_plan(role, result, snapshot, builder, policy, variant):
    eligibility = builder.interactions.view(snapshot, builder.registry, builder.store)
    if role in {"intra", "generator"}:
        validate_intra(result, snapshot, eligibility, builder.store, owner=role)
    elif role == "inter":
        validate_inter(result, snapshot, eligibility, builder.store, policy, variant)
    else:
        validate_intra(result.intra, snapshot, eligibility, builder.store, owner="joint")
        validate_inter(
            result.inter, snapshot, eligibility, builder.store, policy, variant, owner="joint"
        )
