from types import SimpleNamespace

from user_simulator.config import GenerationSettings, load_model_profile
from user_simulator.controller.llm_controller import LLMController
from user_simulator.domain.dag import DagNode
from user_simulator.domain.enums import Difficulty, SatisfactionLevel
from user_simulator.domain.results import ControllerResult
from user_simulator.domain.state import EpisodeState
from user_simulator.llm.mock import MockStructuredLLMClient
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient


async def test_controller_retries_semantically_invalid_output() -> None:
    client = MockStructuredLLMClient(
        [
            {
                "decisions": [{"node_id": "N2", "exposable": True, "reason": "missing N3"}],
                "end_reachable": False,
                "summary": "invalid",
            },
            {
                "decisions": [
                    {"node_id": "N2", "exposable": True, "reason": "supported"},
                    {"node_id": "N3", "exposable": False, "reason": "not supported"},
                ],
                "end_reachable": False,
                "summary": "valid",
            },
        ]
    )
    controller = LLMController(client, GenerationSettings(temperature=0, max_completion_tokens=100))
    state = EpisodeState(
        sample_id="x",
        difficulty=Difficulty.EASY,
        current_frontier="N1",
        exposed_nodes=["N1"],
        satisfaction={"N1": SatisfactionLevel.UNSATISFIED},
        random_seed=1,
    )
    result = await controller.decide(
        history=[],
        latest_assistant_response="response",
        state=state,
        candidates=[
            DagNode(node_id="N2", node_type="intent", node_intent="two"),
            DagNode(node_id="N3", node_type="intent", node_intent="three"),
        ],
        has_end_edge=False,
    )
    assert [item.node_id for item in result.decisions] == ["N2", "N3"]
    assert len(client.calls) == 2
    assert "Correction required" in client.calls[1]["messages"][-1]["content"]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs = {}

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        content = "" if self.calls == 1 else '{"decisions":[],"end_reachable":false,"summary":"ok"}'
        return SimpleNamespace(
            id=f"request-{self.calls}",
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        )


async def test_openrouter_client_retries_empty_structured_output() -> None:
    completions = _FakeCompletions()
    fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    profile = load_model_profile(
        overrides={
            "retry": {
                "max_attempts": 2,
                "initial_backoff_seconds": 0,
                "maximum_backoff_seconds": 0,
            }
        }
    )
    client = OpenRouterStructuredClient(profile, client=fake_sdk)
    result = await client.generate_structured(
        messages=[{"role": "user", "content": "return an empty decision list"}],
        response_model=ControllerResult,
        schema_name="controller_result",
        generation=GenerationSettings(temperature=0, max_completion_tokens=50),
    )
    assert result.summary == "ok"
    assert completions.calls == 2
    assert completions.last_kwargs["max_tokens"] == 50
    assert "max_completion_tokens" not in completions.last_kwargs
    assert client.last_call_metadata["transport_retry_count"] == 1
