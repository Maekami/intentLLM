"""Synthetic JSON-schema/local-contract agreement for finite action choices."""

import pytest
from gp_helpers import current_plan, delivery, profile
from test_gp_contracts_v2 import normalized

from assistant.goal_progression.contracts import output_schema, validate_inter, validate_intra
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.schemas import InterPlan, IntraPlan

jsonschema = pytest.importorskip("jsonschema")


@pytest.mark.parametrize(
    "action,body,question,blocked,valid",
    [
        ("advance", True, False, False, True),
        ("advance", False, False, False, False),
        ("advance", True, True, False, False),
        ("revise", True, False, False, True),
        ("clarify", False, True, False, True),
        ("clarify", True, True, False, True),
        ("clarify", False, False, False, False),
        (None, False, False, True, True),
        (None, False, False, False, False),
        (None, True, False, True, False),
    ],
)
def test_current_action_shape_is_legal_by_construction(action, body, question, blocked, valid):
    snapshot, store, eligibility = normalized()
    raw = current_plan(snapshot.model_dump())
    item = raw["items"][0]
    item.update(
        action=action,
        deliveries=[delivery("g0001")] if body else [],
        request={"need_id": "n0001", "question_text": "Which synthetic shape?"}
        if question
        else None,
        blocked_by=["n0001"] if blocked else [],
    )
    wire = output_schema(IntraPlan, snapshot=snapshot, request_eligibility=eligibility)
    jsonschema.Draft202012Validator.check_schema(wire)
    validator = jsonschema.Draft202012Validator(wire)
    assert validator.is_valid(raw) is valid
    model = IntraPlan.model_validate(raw)
    if valid:
        validate_intra(model, snapshot, eligibility, store)
    else:
        with pytest.raises(ContractError):
            validate_intra(model, snapshot, eligibility, store)


@pytest.mark.parametrize(
    "action,gid,body,question,valid",
    [
        ("none", None, False, False, True),
        ("none", "g0002", False, False, False),
        ("elicit", "g0002", False, True, True),
        ("elicit", "g0002", True, True, False),
        ("elicit", "g0002", False, False, False),
        ("anticipate", "g0002", True, False, True),
        ("anticipate", "g0002", True, True, False),
        ("anticipate", None, True, False, False),
        ("anticipate", "g0002", False, False, False),
    ],
)
def test_adjacent_action_shape_is_legal_by_construction(action, gid, body, question, valid):
    snapshot, store, eligibility = normalized()
    raw = {
        "action": action,
        "goal_id": gid,
        "deliveries": [delivery("g0002")] if body else [],
        "request": {"need_id": "n0002", "question_text": "Which synthetic label?"}
        if question
        else None,
    }
    wire = output_schema(InterPlan, snapshot=snapshot, request_eligibility=eligibility)
    jsonschema.Draft202012Validator.check_schema(wire)
    assert jsonschema.Draft202012Validator(wire).is_valid(raw) is valid
    model = InterPlan.model_validate(raw)
    if valid:
        validate_inter(model, snapshot, eligibility, store, profile().goal_progression.policy)
    else:
        with pytest.raises(ContractError):
            validate_inter(model, snapshot, eligibility, store, profile().goal_progression.policy)
