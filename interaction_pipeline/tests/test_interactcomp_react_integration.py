import json
from pathlib import Path

import pytest
from assistant.baselines.interactcomp_action_guard import InteractCompActionGuard
from assistant.baselines.interactcomp_react import (
    InteractCompReActBaseline,
    InteractCompReActSession,
)
from assistant.config import GenerationSettings
from assistant.llm.base import GeneratedResponse
from assistant.prompt import SystemPrompt
from user_simulator.audit.logger import AuditLogger
from user_simulator.config import EnvironmentSettings as SimulatorEnvironmentSettings
from user_simulator.config import load_model_profile as load_simulator_model_profile
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty, SatisfactionLevel
from user_simulator.domain.results import (
    SatisfactionUpdateResult,
    make_node_satisfaction_decision,
)
from user_simulator.engine.episode import Episode
from user_simulator.mock_components import MockController, MockUserRealizer
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy

from interaction_pipeline.config import load_pipeline_config
from interaction_pipeline.core import run_interaction
from interaction_pipeline.interactcomp_guard import SimulatorProfileGuardClient
from interaction_pipeline.prepare import prepare_pipeline

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sample() -> Sample:
    return Sample.model_validate(
        {
            "sample_id": "react-pipeline-fixture",
            "reason_dag": {
                "nodes": [
                    {
                        "node_id": "N1",
                        "node_type": "intent",
                        "surface_user_message": "Please help me.",
                        "node_intent": "Get help.",
                    },
                    {
                        "node_id": "END",
                        "node_type": "terminal",
                        "node_intent": "Done.",
                    },
                ],
                "edges": [{"edge_id": "E1", "source": "N1", "target": "END"}],
            },
        }
    )


def test_batch_preparation_resolves_react_prompt_profile_and_turn_limit() -> None:
    prepared = prepare_pipeline(
        load_pipeline_config(),
        simulator_model_profile="deepseek_v4_flash_0731",
        assistant_model_profile="qwen_3_6_27b_react",
        assistant_baseline="interactcomp_react",
        max_turns=20,
    )

    prompt_path = Path(prepared.assistant_config.prompts.interactcomp_react)
    guard_prompt_path = Path(prepared.assistant_config.prompts.interactcomp_action_guard)
    profile_path = Path(prepared.assistant_config.models["assistant"])
    assert prompt_path.is_absolute()
    assert prompt_path.is_file()
    assert guard_prompt_path.is_absolute()
    assert guard_prompt_path.is_file()
    assert profile_path.is_absolute()
    assert profile_path.name == "qwen_3_6_27b_react.yaml"
    assert prepared.assistant_config.components.baseline == "interactcomp_react"
    assert prepared.simulator_config.policy.max_turns == 20

    snapshot = prepared.batch_config_snapshot(
        batch_id="fixture",
        created_at="2026-08-31T00:00:00+00:00",
        samples=[_sample()],
        concurrency=1,
        update_memory=False,
    )
    assert snapshot["run_overview"]["baseline"] == "interactcomp_react"
    assert snapshot["assistant"]["active_baseline"] == "interactcomp_react"
    assert snapshot["assistant"]["active_prompt"]["name"] == "interactcomp_react"
    assert snapshot["assistant"]["active_prompt"]["path"] == str(prompt_path)
    assert len(snapshot["assistant"]["active_prompt"]["prompt_hash"]) == 64
    assert snapshot["assistant"]["action_guard"]["enabled"] is True
    assert snapshot["assistant"]["action_guard"]["prompt"]["name"] == ("interactcomp_action_guard")
    assert snapshot["assistant"]["action_guard"]["prompt"]["path"] == str(guard_prompt_path)
    guard_snapshot = snapshot["assistant"]["action_guard"]
    assert guard_snapshot["model_profile"]["source"] == "cli.simulator_model_profile"
    assert Path(guard_snapshot["model_profile"]["source_path"]).name == (
        "deepseek_v4_flash_0731.yaml"
    )
    assert guard_snapshot["model_profile"]["profile_name"] == "deepseek_v4_flash_0731"
    assert guard_snapshot["model_profile"]["model_id"] == ("deepseek/deepseek-v4-flash-0731")
    assert guard_snapshot["generation"]["temperature"] == 1.0
    assert guard_snapshot["generation"]["top_p"] == 1.0
    assert guard_snapshot["confirm_rejections"] is False
    assert guard_snapshot["cache_verdicts"] is False
    assert guard_snapshot["invalid_output_retries"] == 0
    assert snapshot["run_overview"]["models"]["action_guard"] == {
        "profile_name": "deepseek_v4_flash_0731",
        "model_id": "deepseek/deepseek-v4-flash-0731",
        "source": "cli.simulator_model_profile",
    }


@pytest.mark.parametrize(
    "assistant_profile",
    (
        "qwen_3_6_27b_react",
        "gpt_5_6_luna_react",
        "gemini_3_6_flash_react",
    ),
)
def test_all_assistant_profiles_use_the_same_explicit_simulator_guard_profile(
    assistant_profile: str,
) -> None:
    prepared = prepare_pipeline(
        load_pipeline_config(),
        simulator_model_profile="deepseek_v4_flash_0731",
        assistant_model_profile=assistant_profile,
        assistant_baseline="interactcomp_react",
    )
    snapshot = prepared.batch_config_snapshot(
        batch_id=assistant_profile,
        created_at="2026-09-02T00:00:00+00:00",
        samples=[_sample()],
        concurrency=1,
        update_memory=False,
    )

    guard = snapshot["assistant"]["action_guard"]["model_profile"]
    assert guard["source"] == "cli.simulator_model_profile"
    assert Path(guard["source_path"]).name == "deepseek_v4_flash_0731.yaml"
    assert guard["model_id"] == "deepseek/deepseek-v4-flash-0731"
    assert snapshot["assistant"]["model_profile"]["settings"]["profile_name"] == (assistant_profile)


def test_hard_mode_keeps_abstract_realizer_but_guard_uses_explicit_yaml_source() -> None:
    prepared = prepare_pipeline(
        load_pipeline_config(overrides={"run": {"difficulty": "hard"}}),
        simulator_model_profile="deepseek_v4_flash_0731",
        assistant_model_profile="qwen_3_6_27b_react",
        assistant_baseline="interactcomp_react",
    )

    guard_path = Path(prepared.action_guard_model_profile_path or "")
    assert guard_path.name == "deepseek_v4_flash_0731.yaml"
    assert prepared.action_guard_model_source == "cli.simulator_model_profile"
    assert {
        Path(prepared.simulator_config.models[key])
        for key in ("controller", "satisfaction", "realizer_clear", "realizer_abstract")
    } == {guard_path}
    profile = load_simulator_model_profile(str(guard_path))
    assert "realizer_abstract" in profile.generation
    assert prepared.config.run.difficulty == "hard"


def test_nonuniform_simulator_config_requires_explicit_profile_for_guard(tmp_path) -> None:
    custom = tmp_path / "nonuniform_simulator.yaml"
    custom.write_text(
        "models:\n"
        "  controller: gpt_5_6_luna\n"
        "  satisfaction: gpt_5_6_luna\n"
        "  realizer_clear: deepseek_v4_flash_0731\n"
        "  realizer_abstract: deepseek_v4_flash_0731\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="--simulator-model-profile"):
        prepare_pipeline(
            load_pipeline_config(),
            simulator_config_path=custom,
            assistant_model_profile="qwen_3_6_27b_react",
            assistant_baseline="interactcomp_react",
        )


class FakeStructuredGuardClient:
    def __init__(self) -> None:
        self.last_call_metadata = {
            "model_id": "deepseek/deepseek-v4-flash-0731",
            "model_profile": "deepseek_v4_flash_0731",
            "request_id": "guard-fixture",
        }
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["response_model"](ok=True, reason="one communication direction")


async def test_guard_client_uses_simulator_profile_with_shared_guard_generation() -> None:
    profile = load_simulator_model_profile(
        str(REPOSITORY_ROOT / "user_simulator/configs/models/deepseek_v4_flash_0731.yaml")
    )
    structured = FakeStructuredGuardClient()
    client = SimulatorProfileGuardClient(
        profile,
        SimulatorEnvironmentSettings(),
        client=structured,
    )

    result = await client.generate(
        messages=[
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": '{"question":"Which city?"}'},
        ],
        generation=GenerationSettings(
            temperature=1.0,
            top_p=1.0,
            max_completion_tokens=2048,
        ),
    )

    assert result.content == '{"ok":true,"reason":"one communication direction"}'
    assert client.profile is profile
    assert client.last_call_metadata["model_profile"] == "deepseek_v4_flash_0731"
    call = structured.calls[0]
    assert call["generation"].temperature == 1.0
    assert call["generation"].top_p == 1.0
    assert call["generation"].max_completion_tokens == 2048


def test_pipeline_switch_disables_guard_and_snapshots_effective_state() -> None:
    prepared = prepare_pipeline(
        load_pipeline_config(),
        assistant_model_profile="qwen_3_6_27b_react",
        assistant_baseline="interactcomp_react",
        react_action_guard=False,
        max_turns=20,
    )

    snapshot = prepared.batch_config_snapshot(
        batch_id="guard-off",
        created_at="2026-09-01T00:00:00+00:00",
        samples=[_sample()],
        concurrency=1,
        update_memory=False,
    )

    assert prepared.assistant_config.interactcomp_react.action_guard.enabled is False
    assert snapshot["assistant"]["action_guard"] == {
        "enabled": False,
        "prompt": None,
        "generation": None,
        "model_profile": None,
        "confirm_rejections": None,
        "cache_verdicts": None,
        "invalid_output_retries": None,
    }


class NeverSatisfiedUpdater:
    def __init__(self) -> None:
        self.last_call_metadata: dict = {}

    async def update(self, *, state, exposed_nodes, **kwargs) -> SatisfactionUpdateResult:
        return SatisfactionUpdateResult(
            updates=[
                make_node_satisfaction_decision(
                    node_id=node.node_id,
                    status=SatisfactionLevel.UNSATISFIED,
                    reason="The fixture intentionally leaves this intent unresolved.",
                    remaining_gap=None,
                )
                for node in exposed_nodes
            ],
            summary="No exposed intent is satisfied in this fixture.",
        )


class AlwaysAnswerClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_call_metadata: dict = {}
        self.profile = type("Profile", (), {"profile_name": "react-fixture"})()

    async def generate(self, *, messages, generation) -> GeneratedResponse:
        self.calls += 1
        self.last_call_metadata = {
            "model_id": "react-fixture",
            "request_id": str(self.calls),
            "input_tokens": 1,
            "output_tokens": 1,
            "thinking_tokens": 0,
            "answer_tokens": 1,
            "latency_seconds": 0.0,
            "transport_retry_count": 0,
        }
        return GeneratedResponse(
            content=(
                '{"action":"answer","params":{"answer":"Here is an answer.","confidence":"80"}}'
            ),
            metadata=dict(self.last_call_metadata),
        )


class RejectingGuardClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_call_metadata: dict = {}

    async def generate(self, *, messages, generation) -> GeneratedResponse:
        self.calls += 1
        self.last_call_metadata = {
            "model_id": "guard-fixture",
            "request_id": f"guard-{self.calls}",
            "input_tokens": 1,
            "output_tokens": 1,
            "thinking_tokens": 0,
            "answer_tokens": 1,
            "latency_seconds": 0.0,
            "transport_retry_count": 0,
        }
        return GeneratedResponse(
            content='{"ok":false,"reason":"answer solicits a user reply"}',
            metadata=dict(self.last_call_metadata),
        )


async def test_twenty_answers_do_not_terminate_or_trigger_a_twenty_first_call(tmp_path) -> None:
    sample = _sample()
    audit = AuditLogger(sample.sample_id, output_dir=tmp_path, run_id="react-turn-limit")
    episode = Episode(
        sample=sample,
        difficulty=Difficulty.EASY,
        seed=42,
        controller=MockController(),
        satisfaction_updater=NeverSatisfiedUpdater(),
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=MockUserRealizer(),
        audit_logger=audit,
        max_turns=20,
    )
    client = AlwaysAnswerClient()
    prompt = SystemPrompt.load(
        REPOSITORY_ROOT / "assistant/configs/prompts/interactcomp_react.yaml"
    )
    assistant = InteractCompReActSession(
        client,
        GenerationSettings(max_completion_tokens=128),
        InteractCompReActBaseline(prompt),
    )

    result = await run_interaction(
        episode=episode,
        assistant=assistant,
        audit=audit,
        update_memory=False,
        stop_before_over_budget_generation=True,
    )

    assert result.status == "failed"
    assert result.turns == 20
    assert result.error is not None
    assert result.error.startswith("EpisodeTurnLimitError:")
    assert client.calls == 20
    assert len(assistant.history) == 40
    assert episode.state.end_exposed is True
    assert episode.state.terminated is False


async def test_exhausted_local_action_budget_is_preserved_in_failure_audit(tmp_path) -> None:
    sample = _sample()
    audit = AuditLogger(sample.sample_id, output_dir=tmp_path, run_id="react-budget")
    episode = Episode(
        sample=sample,
        difficulty=Difficulty.EASY,
        seed=42,
        controller=MockController(),
        satisfaction_updater=NeverSatisfiedUpdater(),
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=MockUserRealizer(),
        audit_logger=audit,
        max_turns=20,
    )
    agent_client = AlwaysAnswerClient()
    guard_client = RejectingGuardClient()
    react_prompt = SystemPrompt.load(
        REPOSITORY_ROOT / "assistant/configs/prompts/interactcomp_react.yaml"
    )
    guard_prompt = SystemPrompt.load(
        REPOSITORY_ROOT / "assistant/configs/prompts/interactcomp_action_guard.yaml"
    )
    guard = InteractCompActionGuard(
        guard_prompt,
        GenerationSettings(
            temperature=1.0,
            top_p=1.0,
            max_completion_tokens=128,
        ),
        guard_client,
    )
    assistant = InteractCompReActSession(
        agent_client,
        GenerationSettings(max_completion_tokens=128),
        InteractCompReActBaseline(
            react_prompt,
            action_guard=guard,
            semantic_action_budget_per_turn=2,
        ),
    )

    result = await run_interaction(
        episode=episode,
        assistant=assistant,
        audit=audit,
        update_memory=False,
    )

    assert result.status == "failed"
    assert result.turns == 0
    assert result.error is not None
    assert result.error.startswith("InteractCompActionBudgetExhaustedError:")
    assert agent_client.calls == 2
    # Official-aligned validation makes one independent call per candidate.
    assert guard_client.calls == 2
    assert assistant.history == ()
    events = [
        json.loads(line) for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    failed = next(event for event in events if event["event_type"] == "assistant_generation_failed")
    metadata = failed["payload"]["llm_call"]
    assert metadata["interactcomp_semantic_action_slots_used"] == 2
    assert metadata["interactcomp_semantic_action_budget_per_turn"] == 2
    assert metadata["interactcomp_semantic_action_budget_exhausted"] is True
    assert metadata["interactcomp_guard_rejection_count"] == 2
    assert "interactcomp_guard_confirmed_rejection_count" not in metadata
    assert metadata["interactcomp_guard_cache_hit_count"] == 0
    transcript = [
        json.loads(line) for line in (result.run_dir / "transcript.jsonl").read_text().splitlines()
    ]
    assert [message["role"] for message in transcript] == ["user"]
    assert not any(event["event_type"] == "assistant_generation_completed" for event in events)
