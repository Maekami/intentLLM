"""Resource defaults must not become fixed experimental budget limits."""

import pytest
from pydantic import ValidationError

from assistant.config import (
    GoalProgressionSettings,
    InteractCompActionGuardSettings,
    InteractCompReActSettings,
    MemoryContextSettings,
    MemoryRetrievalSettings,
    ReMemSettings,
)


@pytest.mark.parametrize(
    "model,field,value",
    [
        (MemoryRetrievalSettings, "query_max_characters", 1),
        (MemoryContextSettings, "max_characters", 1),
        (ReMemSettings, "max_iterations", 101),
        (InteractCompActionGuardSettings, "invalid_output_retries", 11),
        (InteractCompReActSettings, "semantic_action_budget_per_turn", 101),
        (InteractCompReActSettings, "format_retries_per_slot", 21),
    ],
)
def test_resource_budget_is_user_configurable(model, field, value):
    assert getattr(model.model_validate({field: value}), field) == value
    with pytest.raises(ValidationError):
        model.model_validate({field: -1})


@pytest.mark.parametrize("retries,limit,timeout", [(0, 1, 0.1), (3, 32, 300), (2, 64, 600)])
def test_gp_budgets_and_snapshot_preserve_configured_values(retries, limit, timeout):
    settings = GoalProgressionSettings(
        variant="joint",
        prompts={role: f"{role}.yaml" for role in ("tracker", "joint", "generator")},
        format_retries=retries,
        max_in_flight_requests=limit,
        turn_timeout_seconds=timeout,
    )
    raw = settings.model_dump(mode="json")
    assert raw["format_retries"] == retries
    assert raw["max_in_flight_requests"] == limit
    assert raw["turn_timeout_seconds"] == timeout
    for field, value in [
        ("format_retries", -1), ("format_retries", 4),
        ("max_in_flight_requests", 0), ("turn_timeout_seconds", 0),
    ]:
        with pytest.raises(ValidationError):
            GoalProgressionSettings.model_validate({**raw, field: value})
