"""Wire compilation preserves who writes instructions versus visible text."""

import pytest

from assistant.goal_progression.contracts import output_schema
from assistant.goal_progression.schemas import (
    CurrentRealization,
    DeliveryProposal,
    InterPlan,
    IntraPlan,
    JointPlan,
    RequestProposal,
)


@pytest.mark.parametrize("model", [IntraPlan, InterPlan, JointPlan, CurrentRealization])
def test_planner_wire_keeps_distinct_delivery_and_request_field_contracts(model):
    schema = output_schema(model)
    definitions = schema["$defs"]
    for owner, field in ((DeliveryProposal, "target"), (RequestProposal, "question_text")):
        expected = owner.model_json_schema()["properties"][field]["description"]
        assert definitions[owner.__name__]["properties"][field]["description"] == expected
    target = definitions["DeliveryProposal"]["properties"]["target"]["description"]
    question = definitions["RequestProposal"]["properties"]["question_text"]["description"]
    assert target != question
