from types import SimpleNamespace

import pytest

from assistant.config import EnvironmentSettings, load_model_profile
from assistant.exceptions import InvalidModelResponseError, OpenRouterRequestError
from assistant.llm import openrouter_client as client_module
from assistant.llm.openrouter_client import OpenAICompatibleChatClient, OpenRouterChatClient


class FakeCompletions:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def response(
    content: str = "answer",
    *,
    choices: bool = True,
    refusal: str | None = None,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    prompt_tokens: int = 2,
    completion_tokens: int = 3,
    reasoning_tokens: int | None = None,
):
    choice_values = (
        [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    refusal=refusal,
                    reasoning_content=reasoning_content,
                    reasoning_details=None,
                ),
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


def client_with(profile_name: str, responses: list):
    completions = FakeCompletions(responses)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile(
        profile_name,
        overrides={
            "retry": {
                "max_attempts": len(responses),
                "initial_backoff_seconds": 0,
                "maximum_backoff_seconds": 0,
            }
        },
    )
    return OpenRouterChatClient(profile, client=sdk), completions, profile


async def test_luna_request_omits_temperature_and_sends_reasoning() -> None:
    client, completions, profile = client_with("gpt_5_6_luna", [response()])
    result = await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )
    assert result.content == "answer"
    request = completions.calls[0]
    assert request["model"] == "openai/gpt-5.6-luna"
    assert request["max_tokens"] == 32768
    assert "temperature" not in request
    assert request["extra_body"]["provider"]["require_parameters"] is True
    assert request["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "max",
        "exclude": True,
    }


async def test_luna_non_thinking_uses_explicit_none_effort() -> None:
    client, completions, profile = client_with(
        "gpt_5_6_luna_non_thinking",
        [response()],
    )
    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )

    request = completions.calls[0]
    assert request["model"] == "openai/gpt-5.6-luna"
    assert request["max_tokens"] == 32768
    assert "temperature" not in request
    assert request["extra_body"]["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert request["extra_body"]["reasoning"] == {"effort": "none"}


async def test_usage_records_thinking_and_answer_tokens_without_reasoning_text() -> None:
    client, _, profile = client_with(
        "gpt_5_6_luna",
        [response(prompt_tokens=11, completion_tokens=17, reasoning_tokens=7)],
    )
    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )

    assert client.last_call_metadata["input_tokens"] == 11
    assert client.last_call_metadata["output_tokens"] == 17
    assert client.last_call_metadata["thinking_tokens"] == 7
    assert client.last_call_metadata["answer_tokens"] == 10
    assert "reasoning_content" not in client.last_call_metadata
    assert "reasoning_details" not in client.last_call_metadata


async def test_usage_keeps_token_breakdown_unknown_when_provider_omits_it() -> None:
    client, _, profile = client_with("qwen_3_6_27b", [response(completion_tokens=9)])
    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )

    assert client.last_call_metadata["output_tokens"] == 9
    assert client.last_call_metadata["thinking_tokens"] is None
    assert client.last_call_metadata["answer_tokens"] is None


async def test_deepseek_v4_flash_request_uses_pinned_model_and_high_reasoning() -> None:
    client, completions, profile = client_with("deepseek_v4_flash_0731", [response()])
    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )
    request = completions.calls[0]
    assert request["model"] == "deepseek/deepseek-v4-flash-0731"
    assert request["temperature"] == 1.0
    assert request["max_tokens"] == 8192
    assert request["extra_body"]["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert request["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "high",
        "exclude": True,
    }


async def test_qwen_request_sends_configured_temperature_without_effort() -> None:
    client, completions, profile = client_with("qwen_3_6_27b", [response()])
    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )
    request = completions.calls[0]
    assert request["model"] == "qwen/qwen3.6-27b"
    assert request["temperature"] == 1.0
    assert request["top_p"] == 0.95
    assert request["presence_penalty"] == 0.0
    assert request["extra_body"]["top_k"] == 20
    assert request["extra_body"]["min_p"] == 0.0
    assert request["extra_body"]["repetition_penalty"] == 1.0
    assert request["extra_body"]["reasoning"] == {"enabled": True, "exclude": False}
    assert "chat_template_kwargs" not in request["extra_body"]


async def test_qwen_non_thinking_sends_official_sampling_and_disables_reasoning() -> None:
    client, completions, profile = client_with("qwen_3_6_27b_non_thinking", [response()])
    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )
    request = completions.calls[0]
    assert request["temperature"] == 0.7
    assert request["top_p"] == 0.8
    assert request["presence_penalty"] == 1.5
    assert request["extra_body"]["reasoning"] == {"enabled": False}


async def test_local_qwen_uses_chat_template_switch_and_preserves_thinking() -> None:
    completions = FakeCompletions([response(reasoning_content="hidden work")])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile("qwen_3_6_27b_vllm")
    client = OpenRouterChatClient(profile, client=sdk)
    generated = await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )
    request = completions.calls[0]
    assert "provider" not in request["extra_body"]
    assert "reasoning" not in request["extra_body"]
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert generated.content == "answer"
    assert generated.reasoning_content == "hidden work"


async def test_local_vllm_normalizes_multiturn_reasoning_field() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile("qwen_3_6_27b_vllm")
    client = OpenAICompatibleChatClient(profile, client=sdk)

    await client.generate(
        messages=[
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "visible",
                "reasoning_content": "hidden",
            },
            {"role": "user", "content": "second"},
        ],
        generation=profile.generation["assistant"],
    )

    historical_assistant = completions.calls[0]["messages"][1]
    assert historical_assistant["reasoning"] == "hidden"
    assert "reasoning_content" not in historical_assistant


async def test_local_qwen_non_thinking_uses_official_chat_template_switch() -> None:
    completions = FakeCompletions([response()])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile("qwen_3_6_27b_vllm_non_thinking")
    client = OpenAICompatibleChatClient(profile, client=sdk)

    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )

    request = completions.calls[0]
    assert request["model"] == "qwen3.6-27b"
    assert request["temperature"] == 0.7
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "provider" not in request["extra_body"]
    assert "reasoning" not in request["extra_body"]


def test_local_vllm_client_needs_no_openrouter_key(monkeypatch) -> None:
    constructed: dict = {}

    def fake_openai(**kwargs):
        constructed.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(client_module, "AsyncOpenAI", fake_openai)
    profile = load_model_profile("qwen_3_6_27b_vllm")

    OpenAICompatibleChatClient(
        profile,
        EnvironmentSettings(openrouter_api_key="", vllm_api_key=""),
    )

    assert constructed["api_key"] == "EMPTY"
    assert constructed["base_url"] == "http://127.0.0.1:8001/v1"
    assert constructed["default_headers"] == {}


def test_openrouter_client_still_requires_openrouter_key() -> None:
    profile = load_model_profile("qwen_3_6_27b")

    with pytest.raises(OpenRouterRequestError, match="OPENROUTER_API_KEY"):
        OpenAICompatibleChatClient(
            profile,
            EnvironmentSettings(openrouter_api_key=""),
        )


@pytest.mark.parametrize(
    "bad_response",
    [
        response(choices=False),
        response(content=""),
        response(refusal="no"),
        response(finish_reason="length"),
    ],
)
async def test_invalid_text_responses_exhaust_retries(bad_response) -> None:
    client, completions, profile = client_with(
        "gpt_5_6_luna",
        [bad_response, bad_response],
    )
    with pytest.raises(InvalidModelResponseError):
        await client.generate(
            messages=[{"role": "user", "content": "hello"}],
            generation=profile.generation["assistant"],
        )
    assert len(completions.calls) == 2
