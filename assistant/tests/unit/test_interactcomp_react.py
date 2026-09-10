from __future__ import annotations

import json
from typing import Any

import pytest

from assistant.baselines.interactcomp_action_guard import (
    InteractCompActionGuard,
    parse_guard_decision,
)
from assistant.baselines.interactcomp_react import (
    InteractCompActionBudgetExhaustedError,
    InteractCompActionFormatError,
    InteractCompReActBaseline,
    InteractCompReActSession,
    InvalidInteractCompActionError,
    parse_interactcomp_action,
)
from assistant.config import GenerationSettings, load_config, load_model_profile
from assistant.exceptions import ConfigurationError
from assistant.factory import build_assistant_components, configured_components_require_openrouter
from assistant.llm.base import GeneratedResponse
from assistant.prompt import SystemPrompt


class FakeClient:
    def __init__(self, responses: list[GeneratedResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, Any]]] = []
        self.generations: list[GenerationSettings] = []
        self.last_call_metadata: dict[str, Any] = {}

    async def generate(self, *, messages, generation) -> GeneratedResponse:
        self.requests.append(messages)
        self.generations.append(generation)
        generated = self.responses.pop(0)
        self.last_call_metadata = dict(generated.metadata or {})
        return generated


def _generated(
    content: str,
    *,
    request_id: str,
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> GeneratedResponse:
    return GeneratedResponse(
        content=content,
        metadata={
            "model_id": "fixture-model",
            "model_profile": "fixture-profile",
            "provider": "fixture",
            "request_id": request_id,
            "finish_reason": "stop",
            "latency_seconds": 0.25,
            "transport_retry_count": 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": 0,
            "answer_tokens": output_tokens,
            "reasoning_preserved": False,
            "messages": [],
            "response": content,
        },
    )


def _session(
    client: FakeClient,
    *,
    guard_enabled: bool = False,
    guard_client: FakeClient | None = None,
    semantic_action_budget_per_turn: int = 10,
    format_retries_per_slot: int = 3,
    confirm_rejections: bool = False,
    cache_verdicts: bool = False,
    invalid_output_retries: int = 0,
) -> InteractCompReActSession:
    prompt = SystemPrompt.load("configs/prompts/interactcomp_react.yaml")
    action_guard = None
    if guard_enabled:
        action_guard = InteractCompActionGuard(
            SystemPrompt.load("configs/prompts/interactcomp_action_guard.yaml"),
            GenerationSettings(max_completion_tokens=256),
            guard_client or client,
            confirm_rejections=confirm_rejections,
            cache_verdicts=cache_verdicts,
            invalid_output_retries=invalid_output_retries,
        )
    baseline = InteractCompReActBaseline(
        prompt,
        action_guard=action_guard,
        semantic_action_budget_per_turn=semantic_action_budget_per_turn,
        format_retries_per_slot=format_retries_per_slot,
    )
    generation = GenerationSettings(max_completion_tokens=1024)
    return InteractCompReActSession(client, generation, baseline)


def test_prompt_defines_mutually_exclusive_non_terminal_action_boundary() -> None:
    system = SystemPrompt.load("configs/prompts/interactcomp_react.yaml").system
    normalized = " ".join(system.split())

    assert (
        "You are an intelligent agent, designed to answer user's question. In each round, "
        "you can execute one action, and you can get the action's result as observation. "
        "You should think step by step, and output the action you want to execute." in normalized
    )
    assert "Before answering, you MUST:" in system
    assert "Identify ALL missing information dimensions" in system
    assert "Systematically gather evidence for each dimension" in system
    assert "Verify key assumptions through multiple sources/questions" in system
    assert "Most questions have hidden complexities." in system
    assert "mutually exclusive communication purposes" in system
    assert (
        "When the ask action is available, ask a single, specific question about one "
        "attribute, relationship, time, place, or a single yes/no proposition." in normalized
    )
    assert (
        "Do not combine multiple sub-questions or multiple dimensions in one question."
        in normalized
    )
    assert (
        "Description: Ask a single, specific question about one attribute, relationship, "
        "time, place, or a single yes/no proposition. Do not ask for full lists, "
        "summaries, or multiple sub-questions." in normalized
    )
    assert (
        "Ask a single, limited question (one attribute/relationship/time/place or a "
        "yes/no proposition). Do not request all facts or summaries, and do not combine "
        "multiple sub-questions." in normalized
    )
    assert "Do not put a clarification request inside answer" in system
    assert "Non-terminal is only an episode-control property" in system
    assert "There is no separate probing budget" in system
    assert "There is no required minimum or maximum number of ask actions" not in system
    assert "Do not use Markdown fences" not in system
    assert "could materially change a helpful response" not in system
    assert '"action": ""' in system
    assert "Before output, you should think step by step." in system


def test_action_guard_prompt_is_boundary_only_and_preserves_answer_exceptions() -> None:
    system = SystemPrompt.load("configs/prompts/interactcomp_action_guard.yaml").system
    normalized = " ".join(system.split())

    assert "Dialogue is deliberately absent" in system
    assert "task relevance, novelty, redundancy" in system
    assert "judge the candidate question alone" in system
    assert "You are a strict checker for whether a question is allowed." in normalized
    assert (
        "Allow ONLY if the question: - Asks about exactly one attribute, relationship, "
        "time, place, or a single yes/no proposition. - Does NOT ask for all information, "
        "complete descriptions, summaries, or overviews. - Does NOT contain multiple "
        "sub-questions or multiple dimensions in one question." in normalized
    )
    assert "use dialogue only to identify the intended respondent" in system
    assert "Question-shaped text is allowed" in system
    assert "it is invalid only when the" in system
    assert "user is expected to respond" in system
    assert '{"ok": true/false, "reason": "<short reason>"}' in system


def test_action_guard_prompt_does_not_extend_official_ask_scope_semantics() -> None:
    system = SystemPrompt.load("configs/prompts/interactcomp_action_guard.yaml").system

    assert "independent response slots" not in system
    assert "candidate values" not in system
    assert "mutually exclusive alternatives" not in system
    assert "brief acknowledgement" not in system


def test_guard_decision_parser_fails_closed_on_invalid_output() -> None:
    assert parse_guard_decision('{"ok":true,"reason":"valid"}').ok is True
    malformed = parse_guard_decision("not JSON")
    wrong_type = parse_guard_decision('{"ok":"yes","reason":"valid"}')

    assert malformed.ok is False
    assert malformed.valid_output is False
    assert "invalid validator output" in malformed.reason
    assert wrong_type.ok is False
    assert wrong_type.valid_output is False


def test_parser_accepts_reference_json_and_defensive_wrappers() -> None:
    ask = parse_interactcomp_action(
        'preface\n```json\n{"action":"ask","params":{"question":"Which city?"}}\n```'
    )
    answer = parse_interactcomp_action(
        '{"name":"answer","params":{"answer":"Use rail.","confidence":82}}'
    )

    assert ask.name == "ask"
    assert ask.content == "Which city?"
    assert ask.confidence is None
    assert answer.name == "answer"
    assert answer.content == "Use rail."
    assert answer.confidence == "82"


@pytest.mark.parametrize(
    "response",
    (
        '{"action":"search","params":{"query":"x"}}',
        '{"action":"ask","params":{"question":"Q?","extra":"x"}}',
        '{"action":"answer","params":{"answer":"A"}}',
        "not json",
    ),
)
def test_parser_rejects_actions_outside_exact_ask_answer_schema(response: str) -> None:
    with pytest.raises(InvalidInteractCompActionError):
        parse_interactcomp_action(response)


async def test_answer_is_non_terminal_and_the_next_turn_can_choose_ask() -> None:
    client = FakeClient(
        [
            _generated(
                '{"action":"answer","params":{"answer":"Start with option A.","confidence":"70"}}',
                request_id="first",
            ),
            _generated(
                '{"action":"ask","params":{"question":"What is your budget?"}}',
                request_id="second",
            ),
        ]
    )
    session = _session(client)

    first = await session.respond("Help me choose.")
    second = await session.respond("That may not fit.")

    assert first == "Start with option A."
    assert second == "What is your budget?"
    assert [message.content for message in session.history] == [
        "Help me choose.",
        "Start with option A.",
        "That may not fit.",
        "What is your budget?",
    ]
    assert session.last_call_metadata["interactcomp_action"] == "ask"
    assert session.last_call_metadata["interactcomp_answer_is_terminal"] is False
    assert session.last_call_metadata["interactcomp_cumulative_action_counts"] == {
        "answer": 1,
        "ask": 1,
    }
    assert not hasattr(session, "max_turns")
    second_request_history = client.requests[1]
    assert any(
        message.get("role") == "assistant"
        and message.get("content")
        == '{"action":"answer","params":{"answer":"Start with option A.","confidence":"70"}}'
        for message in second_request_history
    )


async def test_invalid_action_retries_are_internal_and_usage_is_aggregated() -> None:
    client = FakeClient(
        [
            _generated("I should ask a question.", request_id="bad", input_tokens=2),
            _generated(
                '```json\n{"action":"ask","params":{"question":"Which region?"}}\n```',
                request_id="good",
                input_tokens=3,
                output_tokens=2,
            ),
        ]
    )
    session = _session(client)

    visible = await session.respond("Recommend something.")

    assert visible == "Which region?"
    assert len(session.history) == 2
    assert "Invalid action" not in client.requests[1][0]["content"]
    retry_feedback = client.requests[1][-1]
    assert retry_feedback["role"] == "user"
    assert "Invalid action. Please choose from: ask, answer." in retry_feedback["content"]
    metadata = session.last_call_metadata
    assert metadata["internal_call_count"] == 2
    assert metadata["interactcomp_invalid_action_count"] == 1
    assert metadata["interactcomp_agent_call_count"] == 2
    assert metadata["interactcomp_action_attempt_count"] == 1
    assert metadata["interactcomp_semantic_action_slots_used"] == 1
    assert metadata["interactcomp_action"] == "ask"
    assert metadata["input_tokens"] == 5
    assert metadata["output_tokens"] == 3
    assert metadata["latency_seconds"] == 0.5
    assert metadata["request_ids"] == ["bad", "good"]
    assert metadata["interactcomp_guard_enabled"] is False
    assert metadata["interactcomp_guard_call_count"] == 0


async def test_enabled_guard_accepts_one_candidate_with_one_extra_call() -> None:
    client = FakeClient(
        [
            _generated(
                '{"action":"answer","params":{"answer":"Use rail.","confidence":"90"}}',
                request_id="agent",
                input_tokens=2,
            ),
            _generated(
                '{"ok":true,"reason":"substantive answer without solicitation"}',
                request_id="guard",
                input_tokens=3,
            ),
        ]
    )
    session = _session(client, guard_enabled=True)

    visible = await session.respond("How should I travel?")

    assert visible == "Use rail."
    assert len(client.requests) == 2
    assert [generation.max_completion_tokens for generation in client.generations] == [
        1024,
        256,
    ]
    assert "strict checker for whether an action payload" in client.requests[1][0]["content"]
    guard_payload = json.loads(client.requests[1][1]["content"])
    assert guard_payload["dialogue"] == [{"role": "user", "content": "How should I travel?"}]
    assert guard_payload["candidate_action"]["action"] == "answer"
    metadata = session.last_call_metadata
    assert metadata["internal_call_count"] == 2
    assert metadata["interactcomp_agent_call_count"] == 1
    assert metadata["interactcomp_guard_call_count"] == 1
    assert metadata["interactcomp_guard_rejection_count"] == 0
    assert metadata["interactcomp_invalid_action_count"] == 0
    assert metadata["input_tokens"] == 5
    assert metadata["request_ids"] == ["agent", "guard"]


async def test_ask_guard_receives_only_the_candidate_question() -> None:
    client = FakeClient(
        [
            _generated(
                '{"action":"ask","params":{"question":"What is your budget?"}}',
                request_id="agent",
            ),
            _generated(
                '{"ok":true,"reason":"The question has one information scope."}',
                request_id="guard",
            ),
        ]
    )
    session = _session(client, guard_enabled=True)

    visible = await session.respond("Help me choose.")

    assert visible == "What is your budget?"
    guard_payload = json.loads(client.requests[1][1]["content"])
    assert guard_payload == {"question": "What is your budget?"}
    assert "dialogue" not in guard_payload
    assert "candidate_action" not in guard_payload


async def test_guard_rejection_retries_candidate_without_committing_it() -> None:
    client = FakeClient(
        [
            _generated(
                '{"action":"answer","params":{"answer":"I can help. What is your budget?",'
                '"confidence":"60"}}',
                request_id="agent-bad",
            ),
            _generated(
                '{"ok":false,"reason":"answer solicits a budget from the current user"}',
                request_id="guard-reject-1",
            ),
            _generated(
                '{"action":"ask","params":{"question":"What is your budget?"}}',
                request_id="agent-good",
            ),
            _generated(
                '{"ok":true,"reason":"one focused request"}',
                request_id="guard-accept",
            ),
        ]
    )
    session = _session(client, guard_enabled=True)

    visible = await session.respond("Help me choose.")

    assert visible == "What is your budget?"
    assert [message.content for message in session.history] == [
        "Help me choose.",
        "What is your budget?",
    ]
    retry_system = client.requests[2][0]["content"]
    retry_feedback = client.requests[2][-1]
    assert "answer solicits a budget" not in retry_system
    assert retry_feedback["role"] == "user"
    assert "Observation: answer_invalid" in retry_feedback["content"]
    assert '"action":"answer"' in retry_feedback["content"]
    assert "answer solicits a budget" not in retry_feedback["content"]
    metadata = session.last_call_metadata
    assert metadata["internal_call_count"] == 4
    assert metadata["interactcomp_agent_call_count"] == 2
    assert metadata["interactcomp_guard_call_count"] == 2
    assert metadata["interactcomp_guard_vote_count"] == 2
    assert metadata["interactcomp_guard_rejection_count"] == 1
    assert metadata["interactcomp_format_rejection_count"] == 0
    assert metadata["interactcomp_invalid_action_count"] == 1
    assert metadata["interactcomp_action"] == "ask"
    assert [item["ok"] for item in metadata["interactcomp_guard_decisions"]] == [
        False,
        True,
    ]
    assert [vote["ok"] for vote in metadata["interactcomp_guard_decisions"][0]["votes"]] == [
        False,
    ]


async def test_opt_in_guard_rejection_confirmation_uses_third_vote_majority() -> None:
    agent = FakeClient(
        [
            _generated(
                '{"action":"ask","params":{"question":"What is your budget?"}}',
                request_id="agent",
            )
        ]
    )
    guard = FakeClient(
        [
            _generated('{"ok":false,"reason":"first rejection"}', request_id="vote-1"),
            _generated('{"ok":true,"reason":"second accepts"}', request_id="vote-2"),
            _generated('{"ok":true,"reason":"third accepts"}', request_id="vote-3"),
        ]
    )
    session = _session(
        agent,
        guard_enabled=True,
        guard_client=guard,
        confirm_rejections=True,
    )

    assert await session.respond("Help me choose.") == "What is your budget?"

    metadata = session.last_call_metadata
    assert metadata["interactcomp_guard_call_count"] == 3
    assert metadata["interactcomp_guard_vote_count"] == 3
    assert metadata["interactcomp_guard_disagreement_count"] == 1
    assert metadata["interactcomp_guard_rejection_count"] == 0
    assert [vote["ok"] for vote in metadata["interactcomp_guard_decisions"][0]["votes"]] == [
        False,
        True,
        True,
    ]


async def test_malformed_guard_output_fails_closed_and_consumes_semantic_slot() -> None:
    agent = FakeClient(
        [
            _generated(
                '{"action":"ask","params":{"question":"What is your budget?"}}',
                request_id="agent-rejected",
            ),
            _generated(
                '{"action":"ask","params":{"question":"Which city?"}}',
                request_id="agent-accepted",
            )
        ]
    )
    guard = FakeClient(
        [
            _generated("invalid", request_id="invalid-verdict"),
            _generated('{"ok":true,"reason":"one dimension"}', request_id="valid-verdict"),
        ]
    )
    session = _session(agent, guard_enabled=True, guard_client=guard)

    assert await session.respond("Help me choose.") == "Which city?"

    metadata = session.last_call_metadata
    assert metadata["interactcomp_guard_call_count"] == 2
    assert metadata["interactcomp_guard_vote_count"] == 2
    assert metadata["interactcomp_guard_invalid_output_count"] == 1
    assert metadata["interactcomp_guard_rejection_count"] == 1
    assert metadata["interactcomp_semantic_action_slots_used"] == 2
    assert metadata["interactcomp_semantic_action_budget_exhausted"] is False
    first = metadata["interactcomp_guard_decisions"][0]
    assert first["status"] == "rejected"
    assert first["ok"] is False
    assert first["validator_attempts"][0]["valid_output"] is False


async def test_disabling_guard_is_exact_prompt_and_schema_only_path() -> None:
    client = FakeClient(
        [
            _generated(
                '{"action":"answer","params":{"answer":"I can help. What is your budget?",'
                '"confidence":"60"}}',
                request_id="agent-only",
            )
        ]
    )
    session = _session(client, guard_enabled=False)

    visible = await session.respond("Help me choose.")

    assert visible == "I can help. What is your budget?"
    assert len(client.requests) == 1
    assert session.last_call_metadata["internal_call_count"] == 1
    assert session.last_call_metadata["interactcomp_guard_enabled"] is False
    assert session.last_call_metadata["interactcomp_guard_call_count"] == 0


async def test_private_rejected_action_memory_persists_across_visible_turns() -> None:
    agent = FakeClient(
        [
            _generated(
                '{"action":"answer","params":{"answer":"What is your budget?","confidence":"50"}}',
                request_id="bad-answer",
            ),
            _generated(
                '{"action":"ask","params":{"question":"What is your budget?"}}',
                request_id="accepted-ask",
            ),
            _generated(
                '{"action":"answer","params":{"answer":"Choose rail.","confidence":"85"}}',
                request_id="next-turn-answer",
            ),
        ]
    )
    guard = FakeClient(
        [
            _generated('{"ok":false,"reason":"solicits user"}', request_id="reject-1"),
            _generated('{"ok":true,"reason":"focused ask"}', request_id="accept-ask"),
            _generated('{"ok":true,"reason":"substantive answer"}', request_id="accept-answer"),
        ]
    )
    session = _session(agent, guard_enabled=True, guard_client=guard)

    assert await session.respond("Help me choose.") == "What is your budget?"
    assert await session.respond("Under $100.") == "Choose rail."

    next_turn_messages = agent.requests[2]
    assert any(
        message.get("role") == "assistant" and '"action":"answer"' in message.get("content", "")
        for message in next_turn_messages
    )
    assert any(
        message.get("role") == "user"
        and "Observation: answer_invalid" in message.get("content", "")
        for message in next_turn_messages
    )
    assert [message.content for message in session.history] == [
        "Help me choose.",
        "What is your budget?",
        "Under $100.",
        "Choose rail.",
    ]
    assert session.last_call_metadata["interactcomp_private_invalid_action_memory_count"] == 1


async def test_tenth_semantic_slot_can_still_accept_an_action() -> None:
    agent_responses = [
        _generated(
            f'{{"action":"ask","params":{{"question":"Budget and city {index}?"}}}}',
            request_id=f"agent-{index}",
        )
        for index in range(1, 10)
    ]
    agent_responses.append(
        _generated(
            '{"action":"ask","params":{"question":"What is your budget?"}}',
            request_id="agent-10",
        )
    )
    guard_responses = [
        _generated(
            '{"ok":false,"reason":"multiple dimensions"}',
            request_id=f"reject-{index}",
        )
        for index in range(1, 10)
    ]
    guard_responses.append(
        _generated('{"ok":true,"reason":"one dimension"}', request_id="accept-10")
    )
    agent = FakeClient(agent_responses)
    guard = FakeClient(guard_responses)
    session = _session(agent, guard_enabled=True, guard_client=guard)

    assert await session.respond("Help.") == "What is your budget?"

    metadata = session.last_call_metadata
    assert metadata["interactcomp_semantic_action_slots_used"] == 10
    assert metadata["interactcomp_guard_rejection_count"] == 9
    assert metadata["interactcomp_semantic_action_budget_exhausted"] is False
    assert metadata["interactcomp_action"] == "ask"


async def test_ten_semantic_rejections_exhaust_local_budget_without_visible_turn() -> None:
    agent = FakeClient(
        [
            _generated(
                '{"action":"ask","params":{"question":"Budget and city?"}}',
                request_id=f"agent-{index}",
            )
            for index in range(10)
        ]
    )
    guard = FakeClient(
        [
            _generated(
                '{"ok":false,"reason":"ask combines two dimensions"}',
                request_id=f"guard-{index}",
            )
            for index in range(10)
        ]
    )
    session = _session(agent, guard_enabled=True, guard_client=guard)

    with pytest.raises(InteractCompActionBudgetExhaustedError, match="10 semantic"):
        await session.respond("Help.")

    assert session.history == ()
    assert len(agent.requests) == 10
    assert len(guard.requests) == 10
    metadata = session.last_call_metadata
    assert metadata["internal_call_count"] == 20
    assert metadata["interactcomp_agent_call_count"] == 10
    assert metadata["interactcomp_guard_call_count"] == 10
    assert metadata["interactcomp_guard_vote_count"] == 10
    assert metadata["interactcomp_guard_cache_hit_count"] == 0
    assert metadata["interactcomp_guard_rejection_count"] == 10
    assert metadata["interactcomp_semantic_action_slots_used"] == 10
    assert metadata["interactcomp_semantic_action_budget_exhausted"] is True
    assert metadata["interactcomp_invalid_action_count"] == 10
    assert metadata["interactcomp_action"] is None


async def test_four_invalid_actions_fail_without_committing_a_dialogue_turn() -> None:
    client = FakeClient([_generated("invalid", request_id=str(index)) for index in range(4)])
    session = _session(client)

    with pytest.raises(InteractCompActionFormatError, match="after 4 attempts"):
        await session.respond("Help.")

    assert session.history == ()
    assert len(client.requests) == 4
    assert session.last_call_metadata["internal_call_count"] == 4
    assert session.last_call_metadata["interactcomp_invalid_action_count"] == 4
    assert session.last_call_metadata["interactcomp_action"] is None


def test_factory_selects_the_dedicated_react_session() -> None:
    config = load_config(
        cli_overrides={
            "components": {"baseline": "interactcomp_react"},
            "models": {"assistant": "qwen_3_6_27b_react"},
        }
    )
    client = FakeClient([])
    guard_client = FakeClient([])

    components = build_assistant_components(
        config=config,
        client=client,
        action_guard_client=guard_client,
        action_guard_model={
            "source": "fixture.simulator_model_profile",
            "profile_name": "fixture-simulator",
        },
    )

    assert isinstance(components.baseline, InteractCompReActBaseline)
    assert isinstance(components.session, InteractCompReActSession)
    assert components.prompt is not None
    assert components.prompt.name == "interactcomp_react"
    assert components.memory_framework is None
    assert components.skill_framework is None
    assert components.baseline.action_guard is not None
    assert components.baseline.action_guard.client is guard_client
    assert components.action_guard_prompt is not None
    assert components.action_guard_prompt.name == "interactcomp_action_guard"
    assert components.action_guard_generation == (config.interactcomp_react.action_guard.generation)
    assert components.action_guard_model == {
        "source": "fixture.simulator_model_profile",
        "profile_name": "fixture-simulator",
    }


def test_factory_switch_disables_the_second_layer() -> None:
    config = load_config(
        cli_overrides={
            "components": {"baseline": "interactcomp_react"},
            "models": {"assistant": "qwen_3_6_27b_react"},
            "interactcomp_react": {"action_guard": {"enabled": False}},
        }
    )

    components = build_assistant_components(config=config, client=FakeClient([]))

    assert isinstance(components.baseline, InteractCompReActBaseline)
    assert components.baseline.action_guard is None
    assert components.action_guard_prompt is None
    assert components.action_guard_generation is None
    assert components.action_guard_model is None


def test_factory_never_silently_falls_back_to_assistant_client_for_guard() -> None:
    config = load_config(
        cli_overrides={
            "components": {"baseline": "interactcomp_react"},
            "models": {"assistant": "qwen_3_6_27b_react"},
        }
    )

    with pytest.raises(ConfigurationError, match="--simulator-model-profile"):
        build_assistant_components(config=config, client=FakeClient([]))


@pytest.mark.parametrize(
    ("react_profile_name", "comparison_profile_name"),
    (
        ("qwen_3_6_27b_react", "qwen_3_6_27b_exprag"),
        ("gpt_5_6_luna_react", "gpt_5_6_luna_exprag"),
        ("gemini_3_6_flash_react", "gemini_3_6_flash_exprag"),
    ),
)
def test_react_profiles_align_model_and_generation_settings(
    react_profile_name: str,
    comparison_profile_name: str,
) -> None:
    react = load_model_profile(react_profile_name)
    comparison = load_model_profile(comparison_profile_name)

    assert react.provider == comparison.provider
    assert react.model_id == comparison.model_id
    assert react.base_url == comparison.base_url
    assert react.routing == comparison.routing
    assert react.reasoning == comparison.reasoning
    assert react.generation["assistant"] == comparison.generation["assistant"]
    assert set(react.generation) == {"assistant"}
    assert react.retry == comparison.retry
    assert react.memory is None
    assert react.skill is None


def test_react_guard_sampling_is_shared_in_baseline_config_not_assistant_profiles() -> None:
    config = load_config()
    guard_generation = config.interactcomp_react.action_guard.generation

    assert (guard_generation.temperature, guard_generation.top_p) == (1.0, 1.0)
    assert guard_generation.max_completion_tokens == 2048
    for profile_name in (
        "qwen_3_6_27b_react",
        "gpt_5_6_luna_react",
        "gemini_3_6_flash_react",
    ):
        assert set(load_model_profile(profile_name).generation) == {"assistant"}


def test_react_rejects_memory_or_static_skill_stacking() -> None:
    memory_config = load_config(
        cli_overrides={
            "components": {"baseline": "interactcomp_react"},
            "models": {"assistant": "qwen_3_6_27b_exprag"},
        }
    )
    skill_config = load_config(
        cli_overrides={
            "components": {
                "baseline": "interactcomp_react",
                "profile_skill_enabled": False,
            },
            "models": {"assistant": "qwen_3_6_27b_trace2skill"},
        }
    )

    with pytest.raises(ConfigurationError, match="memory-free"):
        configured_components_require_openrouter(memory_config)
    with pytest.raises(ConfigurationError, match="cannot be combined with a static skill"):
        configured_components_require_openrouter(skill_config)
