import os

import pytest

from user_simulator.config import EnvironmentSettings, GenerationSettings, load_model_profile
from user_simulator.domain.results import (
    ControllerResult,
    SatisfactionUpdateResult,
    UserGenerationResult,
)
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY is not set",
    ),
]


@pytest.mark.parametrize(
    ("schema", "name", "instruction"),
    [
        (
            ControllerResult,
            "controller_live",
            'Return decisions=[] and summary="ok".',
        ),
        (
            SatisfactionUpdateResult,
            "satisfaction_live",
            'Return updates=[], summary="ok".',
        ),
        (
            UserGenerationResult,
            "realizer_live",
            """Return user_message="Hello", selected_node_ids=["N1"],
realization_mode="clear", coverage=[{"node_id":"N1","covered":true}],
contains_unsupported_task_content=false, summary="ok".""",
        ),
    ],
)
async def test_strict_structured_output(schema, name, instruction) -> None:
    profile = load_model_profile()
    client = OpenRouterStructuredClient(profile, EnvironmentSettings())
    result = await client.generate_structured(
        messages=[
            {"role": "system", "content": "Return only the requested structured result."},
            {"role": "user", "content": instruction},
        ],
        response_model=schema,
        schema_name=name,
        generation=GenerationSettings(temperature=0, max_completion_tokens=200),
    )
    assert isinstance(result, schema)
    assert client.last_call_metadata["structured_validation_status"] == "valid"
    assert client.last_call_metadata["require_parameters"] is True
    assert "input_tokens" in client.last_call_metadata
    assert "output_tokens" in client.last_call_metadata
    assert "thinking_tokens" in client.last_call_metadata
    assert "answer_tokens" in client.last_call_metadata
