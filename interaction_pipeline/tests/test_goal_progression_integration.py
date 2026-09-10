"""GP artifacts pass through the real pipeline with local scripted components."""

import json
from pathlib import Path

import pytest
import yaml
from assistant.config import load_model_profile
from assistant.exceptions import ModelRequestError
from assistant.goal_progression import GoalProgressionBaseline, GoalProgressionSession
from assistant.llm.base import GeneratedResponse
from user_simulator.audit.logger import AuditLogger
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.mock_components import MockController, MockSatisfactionUpdater, MockUserRealizer
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy

from interaction_pipeline.config import load_pipeline_config
from interaction_pipeline.core import run_interaction
from interaction_pipeline.prepare import prepare_pipeline

ROOT = Path(__file__).resolve().parents[2]


def sample():
    return Sample.model_validate(
        {
            "sample_id": "gp-fixture",
            "reason_dag": {
                "nodes": [
                    {
                        "node_id": "N1",
                        "node_type": "intent",
                        "surface_user_message": "Please help me.",
                        "node_intent": "PRIVATE_INTENT",
                    },
                    {"node_id": "END", "node_type": "terminal", "node_intent": "Done."},
                ],
                "edges": [{"edge_id": "E1", "source": "N1", "target": "END"}],
            },
        }
    )


class Client:
    def __init__(self, fail=False, tracker_output=None):
        self.messages = []
        self.fail = fail
        self.tracker_output = tracker_output

    async def generate(self, *, messages, generation, response_schema=None):
        self.messages.append(messages)
        budget = generation.max_completion_tokens
        if budget == 4096:
            content = self.tracker_output or json.dumps(
                {"version": 1, "goals": [], "constraints": [], "changed_goal_ids": []}
            )
        elif budget == 2048:
            content = '{"items":[]}'
        elif budget == 1024:
            content = '{"action":"none","goal_id":null,"target":null,"anchor_goal_ids":[],"grounding":null}'
        elif budget == 3072:
            content = '{"intra":{"items":[]},"inter":{"action":"none","goal_id":null,"target":null,"anchor_goal_ids":[],"grounding":null}}'
        elif self.fail:
            raise ModelRequestError("scripted failure")
        else:
            content = "Here is the complete answer."
        return GeneratedResponse(
            content,
            reasoning_content="PRIVATE_REASONING",
            metadata={
                "request_id": str(budget),
                "output_tokens": budget,
                "reasoning_content": "PRIVATE_REASONING",
                "messages": [{"content": "PRIVATE_REASONING"}],
            },
        )

    async def count_visible_tokens(self, content):
        return {"visible_response_tokens": 6, "visible_token_source": "fixture"}


@pytest.mark.parametrize(
    "variant,roles",
    [
        ("full", ["tracker", "intra", "inter", "generator"]),
        ("joint", ["tracker", "joint", "generator"]),
        ("no_tracker", ["intra", "inter", "generator"]),
        ("no_intra", ["tracker", "inter", "generator"]),
        ("no_inter", ["tracker", "intra", "generator"]),
        ("no_anticipate", ["tracker", "intra", "inter", "generator"]),
    ],
)
def test_prepare_batch_and_episode_snapshot(variant, roles, monkeypatch, tmp_path):
    profile_name = "qwen_3_6_27b_gp" + ("" if variant == "full" else "_" + variant)
    prepared = prepare_pipeline(
        load_pipeline_config(),
        assistant_baseline="goal_progression",
        assistant_model_profile=profile_name,
        simulator_model_profile="deepseek_v4_flash_0731",
        max_turns=20,
    )
    snap = prepared.batch_config_snapshot(
        batch_id="gp",
        created_at="2026-09-08",
        samples=[sample()],
        concurrency=1,
        update_memory=False,
    )
    gp = snap["assistant"]["goal_progression"]
    assert gp["variant"] == variant and gp["roles"] == roles
    assert gp["generation"]["generator"]["max_completion_tokens"] == 32768
    assert set(gp["prompts"]) == set(roles)
    assert all(
        p["hash"] and p["system"] and Path(p["path"]).is_absolute() for p in gp["prompts"].values()
    )
    if variant == "joint":
        assert gp["generation"]["joint"]["max_completion_tokens"] == 3072
    # build constructs clients but makes no requests; placeholders prevent env dependency.
    from assistant import factory

    monkeypatch.setattr(factory, "OpenAICompatibleChatClient", lambda *args: Client())
    from types import SimpleNamespace

    from user_simulator.config import load_model_profile as simulator_profile

    import interaction_pipeline.prepare as module

    monkeypatch.setattr(
        module,
        "build_simulator_components",
        lambda **kwargs: SimpleNamespace(
            model_profiles={
                "controller": simulator_profile(
                    str(ROOT / "user_simulator/configs/models/deepseek_v4_flash_0731.yaml")
                )
            },
            controller=MockController(),
            satisfaction_updater=MockSatisfactionUpdater(),
            selection_policy=DifficultySelectionPolicy(),
            realization_policy=DifficultyRealizationPolicy(),
            user_realizer=MockUserRealizer(),
        ),
    )
    built = prepared.build(sample(), seed=42, output_dir=tmp_path, run_id="episode")
    saved = yaml.safe_load((built.audit.run_dir / "config_snapshot.yaml").read_text())
    assert saved["assistant_goal_progression"] == gp


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("level", ["full", "summary"])
@pytest.mark.parametrize("variant", ["full", "no_tracker", "no_intra", "no_inter", "joint", "no_anticipate"])
async def test_role_audit_on_disk_and_private_data_boundary(tmp_path, fail, level, variant):
    item = sample()
    audit = AuditLogger(item.sample_id, output_dir=tmp_path, run_id="gp", level=level)
    episode = Episode(
        sample=item,
        difficulty=Difficulty.EASY,
        seed=42,
        controller=MockController(),
        satisfaction_updater=MockSatisfactionUpdater(),
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=MockUserRealizer(),
        audit_logger=audit,
        max_turns=20,
    )
    name = "qwen_3_6_27b_gp" + ("" if variant == "full" else "_" + variant)
    profile = load_model_profile(str(ROOT / "assistant/configs/models" / (name + ".yaml")))
    client = Client(fail=fail)
    assistant = GoalProgressionSession(
        client, profile.generation["assistant"], GoalProgressionBaseline(), profile=profile
    )
    result = await run_interaction(
        episode=episode, assistant=assistant, audit=audit, update_memory=False
    )
    assert result.status == ("failed" if fail else "completed")
    events_text = (audit.run_dir / "events.jsonl").read_text()
    events = [json.loads(line) for line in events_text.splitlines()]
    kind = "assistant_generation_failed" if fail else "assistant_generation_completed"
    event = next(e for e in events if e["event_type"] == kind)
    metadata = event["payload"]["llm_call"]
    gp = metadata["goal_progression"]
    assert len(gp["calls"]) == len(profile.goal_progression.roles)
    assert metadata["goal_progression"]["committed"] is (not fail)
    assert metadata["visible_response_tokens"] == (0 if fail else 6)
    kinds = [e["event_type"] for e in events]
    for call in gp["calls"]:
        role = call["role"]
        prefix = f"assistant_{role}_"
        requested = kinds.index(prefix + "requested")
        completed = kinds.index(prefix + ("failed" if fail and role == "generator" else "raw_result"))
        assert requested < completed < kinds.index(kind)
        payload = events[completed]["payload"]
        assert payload["role"] == role and payload["attempt"] == 1
        assert payload["usage"] == call["usage"]
        if not (fail and role == "generator"):
            assert payload["raw_output"] == call["raw_output"]
            # Later validation must not mutate the already recorded raw event.
            assert payload["status"] == "received" and "structured_result" not in payload
        if role != "generator":
            validated = kinds.index(prefix + "validated")
            assert completed < validated < kinds.index("assistant_generator_requested")
            assert events[validated]["payload"]["structured_result"] == call["structured_result"]
    assembly = events[kinds.index("assistant_gp_assembly_completed")]["payload"]
    assert assembly["assembly"] == gp["assembly"]
    assert kinds.index("assistant_gp_assembly_completed") < kinds.index("assistant_generator_requested")
    state_kind = "assistant_gp_turn_failed" if fail else "assistant_gp_state_committed"
    committed = events[kinds.index(state_kind)]["payload"]
    assert committed["committed"] is (not fail)
    assert committed["final_reply"] == (None if fail else "Here is the complete answer.")
    assert committed["visible_response_tokens"] == (0 if fail else 6)
    assert "PRIVATE_REASONING" not in events_text
    inputs = json.dumps(client.messages)
    for forbidden in ("N1", "PRIVATE_INTENT", "PRIVATE_REASONING"):
        assert forbidden not in inputs
    for messages in client.messages:
        context = json.loads(messages[0]["content"].split("\n\nRuntime context (JSON):\n", 1)[1])
        observation = json.loads(messages[1]["content"])
        # Static policy may say a signal is NOT satisfaction. The actual runtime
        # input must not contain simulator difficulty or satisfaction information.
        for forbidden in ("difficulty", "satisfaction"):
            assert forbidden not in json.dumps([context, observation])
    transcript = (audit.run_dir / "transcript.jsonl").read_text()
    assert "goal_progression" not in transcript and "changed_goal_ids" not in transcript


async def test_invalid_tracker_attempts_and_fallback_are_immediately_persisted(tmp_path):
    audit = AuditLogger("gp-fixture", output_dir=tmp_path, run_id="fallback")

    class CheckingClient(Client):
        async def generate(self, *, messages, generation, response_schema=None):
            if generation.max_completion_tokens == 32768:
                # Before Generator returns, all configured attempts are already on disk.
                recorded = [json.loads(line) for line in (audit.run_dir / "events.jsonl").read_text().splitlines()]
                kinds = [e["event_type"] for e in recorded]
                assert kinds.count("assistant_tracker_raw_result") == expected_attempts
                assert kinds.count("assistant_tracker_validation_failed") == expected_attempts
                assert "assistant_gp_fallback_applied" in kinds
                assert "assistant_gp_assembly_completed" in kinds
                assert "assistant_intra_requested" not in kinds
            return await super().generate(
                messages=messages, generation=generation, response_schema=response_schema
            )

    profile = load_model_profile(str(ROOT / "assistant/configs/models/qwen_3_6_27b_gp.yaml"))
    expected_attempts = profile.goal_progression.format_retries + 1
    assistant = GoalProgressionSession(
        CheckingClient(tracker_output="Here is advice instead of JSON."),
        profile.generation["assistant"], GoalProgressionBaseline(), profile=profile,
    )
    await assistant.respond("Please help.", audit_sink=lambda kind, payload: audit.log(kind, 1, payload))
    events = [json.loads(line) for line in (audit.run_dir / "events.jsonl").read_text().splitlines()]
    attempts = [e["payload"] for e in events if e["event_type"] == "assistant_tracker_raw_result"]
    assert [p["attempt"] for p in attempts] == list(range(1, expected_attempts + 1))
    assert all(p["raw_output"] == "Here is advice instead of JSON." for p in attempts)
    assert events[-1]["event_type"] == "assistant_gp_state_committed"
    assert events[-1]["payload"]["state"] is None
    assert events[-1]["payload"]["fallbacks"][0]["role"] == "tracker"
