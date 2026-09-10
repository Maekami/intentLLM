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


async def test_gp_schema_is_transmitted_but_generator_remains_unconstrained():
    profile = load_model_profile("qwen_3_6_27b_gp")
    completions = FakeCompletions([response(), response()])
    client = OpenAICompatibleChatClient(
        profile, client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    schema = {"type": "object", "properties": {"version": {"const": 3}}, "required": ["version"], "additionalProperties": False}
    await client.generate(
        messages=[{"role": "user", "content": "Extract state"}],
        generation=profile.generation["tracker"], response_schema=schema,
    )
    await client.generate(
        messages=[{"role": "user", "content": "Write reply"}],
        generation=profile.generation["assistant"],
    )
    assert completions.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "goal_progression_output", "strict": True, "schema": schema},
    }
    assert "response_format" not in completions.calls[1]


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
        "effort": "high",
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


@pytest.mark.parametrize(
    ("profile_name", "effort"),
    [
        ("gemini_3_6_flash", "high"),
        ("gemini_3_6_flash_non_thinking", "minimal"),
    ],
)
async def test_gemini_profiles_send_mandatory_thinking_effort(
    profile_name: str,
    effort: str,
) -> None:
    client, completions, profile = client_with(profile_name, [response()])
    await client.generate(
        messages=[{"role": "user", "content": "hello"}],
        generation=profile.generation["assistant"],
    )

    request = completions.calls[0]
    assert request["model"] == "google/gemini-3.6-flash"
    assert request["max_tokens"] == 65536
    assert "temperature" not in request
    assert "top_p" not in request
    assert request["extra_body"]["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert request["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": effort,
        "exclude": True,
    }


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
    # Exercise the enabled path explicitly; the shipped profile now disables it.
    profile.reasoning.enabled = True
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


async def test_gp_transport_attempts_are_bound_to_response_and_fatal_errors_do_not_retry():
    import httpx
    from openai import APIConnectionError, AuthenticationError

    from assistant.exceptions import ConfigurationError

    class Scripted:
        def __init__(self, values):
            self.values = values
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            value = self.values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

    request = httpx.Request("POST", "http://fixture.invalid/v1/chat/completions")
    transport = Scripted([APIConnectionError(request=request), response()])
    profile = load_model_profile(
        "qwen_3_6_27b_gp",
        overrides={
            "retry": {
                "max_attempts": 2,
                "initial_backoff_seconds": 0,
                "maximum_backoff_seconds": 0,
            }
        },
    )
    client = OpenAICompatibleChatClient(
        profile, client=SimpleNamespace(chat=SimpleNamespace(completions=transport))
    )
    result = await client.generate(
        messages=[{"role": "user", "content": "hello"}], generation=profile.generation["tracker"]
    )
    assert result.metadata["transport_retry_count"] == 1
    assert [a["status"] for a in result.metadata["transport_attempts"]] == ["failed", "success"]
    assert result.metadata["transport_attempts"][0]["usage_status"] == "unknown"
    transport.values = [
        AuthenticationError(
            "unauthorized", response=httpx.Response(401, request=request), body=None
        )
    ]
    with pytest.raises(ConfigurationError) as caught:
        await client.generate(
            messages=[{"role": "user", "content": "hello"}],
            generation=profile.generation["tracker"],
        )
    assert caught.value.call_metadata["transport_attempts"][0]["http_status"] == 401
    assert transport.calls == 3


@pytest.mark.parametrize("nested", [False, True])
async def test_gp_bad_request_preserves_server_message_in_error_and_audit(nested):
    from unittest.mock import AsyncMock

    import httpx
    from openai import BadRequestError

    from assistant.exceptions import ConfigurationError
    from assistant.goal_progression.session import public_usage

    detail = "System message must be at the beginning."
    body = {"message": detail, "type": "BadRequestError"}
    if nested:
        body = {"error": body}
    request = httpx.Request("POST", "http://fixture.invalid/v1/chat/completions")
    create = AsyncMock(
        side_effect=BadRequestError(
            "bad request", response=httpx.Response(400, request=request), body=body
        )
    )
    profile = load_model_profile("qwen_3_6_27b_gp")
    client = OpenAICompatibleChatClient(
        profile,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    with pytest.raises(ConfigurationError, match="HTTP 400") as caught:
        await client.generate(
            messages=[{"role": "user", "content": "hello"}],
            generation=profile.generation["tracker"],
        )
    assert detail in str(caught.value)
    attempt = public_usage(caught.value.call_metadata)["transport_attempts"][0]
    assert attempt["http_status"] == 400
    assert attempt["error_message"] == detail
    assert create.await_count == 1


async def test_visible_tokenizer_counts_only_supplied_text_without_special_tokens(monkeypatch):
    import httpx

    from assistant.exceptions import ModelRequestError

    requests = []

    def handler(request):
        import json

        requests.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"count": 7})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        client_module.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
    )
    profile = load_model_profile("qwen_3_6_27b_gp")
    client = OpenAICompatibleChatClient(profile, client=SimpleNamespace())
    result = await client.count_visible_tokens("the delivered text")
    assert result["visible_response_tokens"] == 7
    assert requests == [
        (
            "http://127.0.0.1:8001/tokenize",
            {
                "model": "qwen3.6-27b",
                "prompt": "the delivered text",
                "add_special_tokens": False,
            },
        )
    ]
    monkeypatch.setattr(
        client_module.httpx,
        "AsyncClient",
        lambda **kwargs: original(
            **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(503))
        ),
    )
    with pytest.raises(ModelRequestError):
        await client.count_visible_tokens("the delivered text")


async def test_truncated_transport_response_retains_usage_without_hidden_reasoning():
    client, _, profile = client_with(
        "qwen_3_6_27b_gp",
        [
            response(
                "unfinished JSON", finish_reason="length", reasoning_content="PRIVATE_REASONING"
            ),
            response("valid"),
        ],
    )
    result = await client.generate(
        messages=[{"role": "user", "content": "hello"}], generation=profile.generation["tracker"]
    )
    attempt = result.metadata["transport_attempts"][0]
    assert attempt["raw_output"] == "unfinished JSON"
    assert attempt["output_tokens"] == 3 and attempt["usage_status"] == "known"
    assert "PRIVATE_REASONING" not in str(attempt)
