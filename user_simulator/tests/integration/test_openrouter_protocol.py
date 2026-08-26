import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from user_simulator.cli import prompt_smoke_test
from user_simulator.config import EnvironmentSettings, GenerationSettings, load_model_profile
from user_simulator.domain.results import ControllerResult, SatisfactionUpdateResult
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
    content: str = '{"decisions":[],"summary":"ok"}',
    *,
    choices: bool = True,
    refusal: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 2,
    completion_tokens: int = 3,
    reasoning_tokens: int | None = None,
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
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=(
                SimpleNamespace(reasoning_tokens=reasoning_tokens)
                if reasoning_tokens is not None
                else None
            ),
        ),
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
    assert request["extra_body"]["plugins"] == [{"id": "response-healing"}]
    assert request["stream"] is False


async def test_luna_profile_omits_unsupported_temperature() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile("gpt_5_6_luna")
    client = OpenRouterStructuredClient(profile, client=sdk)

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=profile.generation["controller"],
    )

    request = completions.calls[0]
    assert profile.model_id == "openai/gpt-5.6-luna"
    assert "temperature" not in request
    assert request["extra_body"]["provider"]["require_parameters"] is True
    assert request["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": profile.generation["controller"].reasoning.effort,
        "exclude": True,
    }
    assert "plugins" not in request["extra_body"]


async def test_disabled_reasoning_omits_effort_and_exclusion() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile("deepseek_v4_pro")
    client = OpenRouterStructuredClient(profile, client=sdk)

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=profile.generation["realizer_clear"],
    )

    assert completions.calls[0]["extra_body"]["reasoning"] == {"enabled": False}
    assert client.last_call_metadata["reasoning"] == {"enabled": False}


async def test_component_reasoning_overrides_profile_fallback() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile(overrides={"reasoning": {"effort": "high"}})
    client = OpenRouterStructuredClient(profile, client=sdk)

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=profile.generation["controller"],
    )

    assert completions.calls[0]["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "low",
        "exclude": True,
    }


async def test_top_level_reasoning_remains_backward_compatible_fallback() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile(overrides={"reasoning": {"enabled": False}})
    client = OpenRouterStructuredClient(profile, client=sdk)

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=GenerationSettings(temperature=0, max_completion_tokens=100),
    )

    assert completions.calls[0]["extra_body"]["reasoning"] == {"enabled": False}


async def test_local_qwen_controller_uses_vllm_chat_template_and_official_sampling() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile("qwen_3_8_27b_vllm")
    client = OpenRouterStructuredClient(profile, client=sdk)

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=profile.generation["controller"],
    )

    request = completions.calls[0]
    assert request["model"] == "Qwen/Qwen3.8-27B"
    assert request["temperature"] == 1.0
    assert request["top_p"] == 0.95
    assert request["presence_penalty"] == 0.0
    assert request["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": False,
            "reasoning_effort": "low",
        },
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    assert request["response_format"]["type"] == "json_schema"
    assert "provider" not in request["extra_body"]
    assert "reasoning" not in request["extra_body"]
    assert "plugins" not in request["extra_body"]
    assert client.last_call_metadata["provider"] == "test-provider"


async def test_local_qwen_realizer_disables_thinking() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile("qwen_3_8_27b_vllm")
    client = OpenRouterStructuredClient(profile, client=sdk)

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=profile.generation["realizer_clear"],
    )

    request = completions.calls[0]
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert client.last_call_metadata["reasoning"] == {"enabled": False}


def test_local_vllm_client_needs_no_openrouter_key(monkeypatch) -> None:
    constructed: dict = {}

    def fake_openai(**kwargs):
        constructed.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "user_simulator.llm.openrouter_client.AsyncOpenAI",
        fake_openai,
    )
    profile = load_model_profile("qwen_3_8_27b_vllm")
    OpenRouterStructuredClient(
        profile,
        EnvironmentSettings(openrouter_api_key="", vllm_api_key=""),
    )

    assert constructed["api_key"] == "EMPTY"
    assert constructed["base_url"] == "http://127.0.0.1:8005/v1"
    assert constructed["default_headers"] == {}


async def test_usage_records_thinking_and_answer_tokens() -> None:
    client, _ = client_with(
        [response(prompt_tokens=11, completion_tokens=17, reasoning_tokens=7)]
    )

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=GenerationSettings(temperature=0, max_completion_tokens=100),
    )

    assert client.last_call_metadata["input_tokens"] == 11
    assert client.last_call_metadata["output_tokens"] == 17
    assert client.last_call_metadata["thinking_tokens"] == 7
    assert client.last_call_metadata["answer_tokens"] == 10


async def test_usage_keeps_breakdown_unknown_when_provider_omits_details() -> None:
    client, _ = client_with([response(completion_tokens=9)])

    await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=GenerationSettings(temperature=0, max_completion_tokens=100),
    )

    assert client.last_call_metadata["output_tokens"] == 9
    assert client.last_call_metadata["thinking_tokens"] is None
    assert client.last_call_metadata["answer_tokens"] is None


@pytest.mark.parametrize(
    "bad_response",
    [
        response(choices=False),
        response(content=""),
        response(content="not-json"),
        response(content='{"decisions":[],"summary":"ok","extra":1}'),
        response(content='{"decisions":[]}'),
        response(refusal="policy refusal"),
        response(finish_reason="length"),
    ],
)
async def test_invalid_response_states_exhaust_retries(bad_response) -> None:
    client, completions = client_with([bad_response, bad_response])
    spec = SCHEMA_REGISTRY["controller_result"]
    with pytest.raises(StructuredOutputError):
        await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            response_model=spec.model,
            schema_name=spec.name,
            generation=GenerationSettings(temperature=0, max_completion_tokens=100),
        )
    assert len(completions.calls) == 2
    assert all(call["response_format"]["type"] == "json_schema" for call in completions.calls)
    assert "STRUCTURED OUTPUT CORRECTION" in completions.calls[1]["messages"][-1]["content"]
    assert client.last_call_metadata["structured_retry_count"] == 1
    assert client.last_call_metadata["transport_retry_count"] == 0
    assert client.last_call_metadata["structured_validation_status"] == "invalid"


async def test_structured_retry_uses_validation_feedback_and_preserves_invalid_output() -> None:
    invalid = {
        "updates": [
            {
                "node_id": "N1",
                "status": "unsatisfied",
                "reason": "The assistant has not yet supplied usable help for this node.",
                "remaining_gap": "This must be null for an unsatisfied status.",
            }
        ],
        "summary": "invalid status and gap combination",
    }
    valid = {
        "updates": [
            {
                "node_id": "N1",
                "status": "unsatisfied",
                "reason": "The assistant has not yet supplied usable help for this node.",
                "remaining_gap": None,
            }
        ],
        "summary": "valid unsatisfied decision",
    }
    client, completions = client_with(
        [response(json.dumps(invalid)), response(json.dumps(valid))]
    )

    result = await client.generate_structured(
        messages=[{"role": "user", "content": "test"}],
        response_model=SatisfactionUpdateResult,
        schema_name="satisfaction_update_result",
        generation=GenerationSettings(temperature=0, max_completion_tokens=100),
    )

    assert result.updates[0].remaining_gap is None
    assert len(completions.calls) == 2
    correction = completions.calls[1]["messages"][-1]["content"]
    assert "STRUCTURED OUTPUT CORRECTION" in correction
    assert "remaining_gap" in correction
    assert completions.calls[1]["messages"][-2] == {
        "role": "assistant",
        "content": json.dumps(invalid),
    }
    assert client.last_call_metadata["structured_retry_count"] == 1
    assert client.last_call_metadata["transport_retry_count"] == 0
    assert client.last_call_metadata["invalid_raw_response"] == invalid


async def test_invalid_enum_is_rejected_without_fallback() -> None:
    payload = {
        "user_message": "hello",
        "selected_node_ids": ["N1"],
        "realization_mode": "unknown",
        "coverage": [{"node_id": "N1", "covered": True}],
        "contains_unsupported_task_content": False,
        "summary": "invalid enum",
    }
    client, completions = client_with(
        [response(json.dumps(payload)), response(json.dumps(payload))]
    )
    spec = SCHEMA_REGISTRY["user_generation_result"]
    with pytest.raises(StructuredOutputError):
        await client.generate_structured(
            messages=[{"role": "user", "content": "test"}],
            response_model=spec.model,
            schema_name=spec.name,
            generation=GenerationSettings(temperature=0, max_completion_tokens=100),
        )
    assert all(call["response_format"]["type"] == "json_schema" for call in completions.calls)


async def test_live_smoke_uses_strict_client_and_actual_component_prompt() -> None:
    record = next(
        item
        for item in prompt_smoke_test._load_cases(Path("tests/fixtures/prompt_cases.jsonl"))
        if item["component"] == "controller"
    )
    completions = FakeCompletions([response(json.dumps(record["fixture_response"]))])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def factory(profile, environment):
        return OpenRouterStructuredClient(profile, environment, client=sdk)

    report = await prompt_smoke_test.run_live_cases(
        [record],
        environment=EnvironmentSettings(openrouter_api_key="fake"),
        client_factory=factory,
    )
    assert report["mode"] == "live"
    assert report["metrics"]["structured_output_valid_rate"] == 1.0
    assert report["metrics"]["semantic_valid_rate"] == 1.0
    assert "reported_coverage_pass_rate" in report["metrics"]
    assert "reported_unsupported_task_content_rate" in report["metrics"]
    assert "reported_unsupported_intent_rate" not in report["metrics"]
    assert "user_coverage_pass_rate" not in report["metrics"]
    request = completions.calls[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["extra_body"]["provider"]["require_parameters"] is True
    assert "VISIBLE CONVERSATION" in request["messages"][-1]["content"]


def _valid_payload(model: type[BaseModel]) -> dict:
    if model is ControllerResult:
        return {"decisions": [], "summary": "valid"}
    if model is SatisfactionUpdateResult:
        return {"updates": [], "summary": "valid"}
    return {
        "user_message": "hello",
        "selected_node_ids": ["N1"],
        "realization_mode": "clear",
        "coverage": [{"node_id": "N1", "covered": True}],
        "contains_unsupported_task_content": False,
        "summary": "valid",
    }
