import pytest

from assistant.baselines.base import BaseBaseline
from assistant.config import GenerationSettings
from assistant.llm.base import GeneratedResponse
from assistant.session import AssistantSession


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []
        self.last_call_metadata = {}

    async def generate(self, *, messages, generation):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return GeneratedResponse(content=response)


class ThinkingFakeClient(FakeClient):
    async def generate(self, *, messages, generation):
        self.calls.append(messages)
        return GeneratedResponse(content="visible", reasoning_content="hidden")


async def test_session_preserves_multiturn_history() -> None:
    client = FakeClient(["first answer", "second answer"])
    session = AssistantSession(
        client,
        GenerationSettings(temperature=0, max_completion_tokens=100),
        BaseBaseline(),
    )
    assert await session.respond("first question") == "first answer"
    assert await session.respond("second question") == "second answer"
    assert client.calls[1] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


async def test_failed_call_does_not_commit_pending_user_message() -> None:
    client = FakeClient([RuntimeError("boom")])
    session = AssistantSession(
        client,
        GenerationSettings(temperature=0, max_completion_tokens=100),
        BaseBaseline(),
    )
    with pytest.raises(RuntimeError, match="boom"):
        await session.respond("not committed")
    assert session.history == ()


async def test_session_retains_reasoning_for_qwen_multiturn_history() -> None:
    client = ThinkingFakeClient([])
    session = AssistantSession(
        client,
        GenerationSettings(temperature=1, max_completion_tokens=100),
        BaseBaseline(),
    )
    await session.respond("first")
    await session.respond("second")
    assert client.calls[1][1] == {
        "role": "assistant",
        "content": "visible",
        "reasoning_content": "hidden",
    }
