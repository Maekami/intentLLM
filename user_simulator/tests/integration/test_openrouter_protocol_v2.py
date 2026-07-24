import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from user_simulator.config import GenerationSettings, load_model_profile
from user_simulator.exceptions import StructuredOutputError
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient
from user_simulator.llm.schema_registry import SCHEMA_REGISTRY


class FakeCompletions:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return response


def response(
    content: str = '{"decisions":[],"end_reachable":false,"summary":"ok"}',
    *,
    choices: bool = True,
    refusal: str | None = None,
    finish_reason: str = "stop",
):
    choice_values = (
        [
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=refusal),
                finish_reason=finish_reason,
            )
        ]
        if choices
        else []
    )
    return SimpleNamespace(
        id="request",
        provider="test-provider",
        choices=choice_values,
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
    )


def client_with(responses: list) -> tuple[OpenRouterStructuredClient, FakeCompletions]:
    completions = FakeCompletions(responses)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile(
        overrides={
            "retry": {
                "max_attempts": len(responses),
                "initial_backoff_seconds": 0,
                "maximum_backoff_seconds": 0,
            }
        }
    )
    return OpenRouterStructuredClient(profile, client=sdk), completions


@pytest.mark.parametrize("spec", list(SCHEMA_REGISTRY.values()))
async def test_exact_strict_request_for_every_schema(spec) -> None:
    sample = _valid_payload(spec.model)
    client, completions = client_with([response(json.dumps(sample))])
    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=spec.model,
        schema_name=spec.name,
        schema_version=spec.version,
        generation=GenerationSettings(temperature=0, max_completion_tokens=100),
    )
    request = completions.calls[0]
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == spec.name
    assert response_format["json_schema"]["schema"]["type"] == "object"
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert request["extra_body"]["provider"]["require_parameters"] is True
    assert request["stream"] is False


@pytest.mark.parametrize(
    "bad_response",
    [
        response(choices=False),
        response(content=""),
        response(content="not-json"),
        response(content='{"decisions":[],"end_reachable":false,"summary":"ok","extra":1}'),
        response(content='{"decisions":[],"end_reachable":false}'),
        response(refusal="policy refusal"),
        response(finish_reason="length"),
    ],
)
async def test_invalid_response_states_exhaust_retries(bad_response) -> None:
    client, completions = client_with([bad_response, bad_response])
    spec = SCHEMA_REGISTRY[("controller_result", 2)]
    with pytest.raises(StructuredOutputError):
        await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            response_model=spec.model,
            schema_name=spec.name,
            schema_version=spec.version,
            generation=GenerationSettings(temperature=0, max_completion_tokens=100),
        )
    assert len(completions.calls) == 2
    assert all(call["response_format"]["type"] == "json_schema" for call in completions.calls)


async def test_invalid_enum_is_rejected_without_fallback() -> None:
    payload = {
        "user_message": "hello",
        "selected_node_ids": [],
        "realization_mode": "unknown",
        "coverage": [],
        "contains_unsupported_intent": False,
        "summary": "invalid enum",
    }
    client, completions = client_with(
        [response(json.dumps(payload)), response(json.dumps(payload))]
    )
    spec = SCHEMA_REGISTRY[("user_generation_result", 2)]
    with pytest.raises(StructuredOutputError):
        await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            response_model=spec.model,
            schema_name=spec.name,
            schema_version=spec.version,
            generation=GenerationSettings(temperature=0, max_completion_tokens=100),
        )
    assert all(call["response_format"]["type"] == "json_schema" for call in completions.calls)


def _valid_payload(model: type[BaseModel]) -> dict:
    if model.__name__ == "ControllerResultV2":
        return {"decisions": [], "end_reachable": False, "summary": "valid"}
    if model.__name__ == "SatisfactionUpdateResultV2":
        return {"updates": [], "summary": "valid"}
    return {
        "user_message": "hello",
        "selected_node_ids": [],
        "realization_mode": "clear",
        "coverage": [],
        "contains_unsupported_intent": False,
        "summary": "valid",
    }
