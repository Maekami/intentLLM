"""GP v2 through the unchanged real pipeline and AuditLogger, with fake model calls."""

import importlib.util
import json
from pathlib import Path

import pytest
from interaction_pipeline.core import run_interaction
from user_simulator.audit.logger import AuditLogger
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.mock_components import MockController, MockSatisfactionUpdater, MockUserRealizer
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy

from assistant.goal_progression.errors import RecoveryExhausted

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "gp_test_helpers", ROOT / "assistant/tests/unit/gp_helpers.py"
)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def events(audit):
    return [json.loads(line) for line in (audit.run_dir / "events.jsonl").read_text().splitlines()]


def episode(audit):
    sample = Sample.model_validate(
        {
            "sample_id": "gp-audit-fixture",
            "reason_dag": {
                "nodes": [
                    {
                        "node_id": "N1",
                        "node_type": "intent",
                        "surface_user_message": "Make a card.",
                        "node_intent": "PRIVATE_INTENT",
                        "reason_text": "Make a card",
                    },
                    {"node_id": "END", "node_type": "terminal", "node_intent": "Done."},
                ],
                "edges": [{"edge_id": "E1", "source": "N1", "target": "END"}],
            },
        }
    )
    return Episode(
        sample=sample,
        difficulty=Difficulty.HARD,
        seed=42,
        controller=MockController(),
        satisfaction_updater=MockSatisfactionUpdater(),
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=MockUserRealizer(),
        audit_logger=audit,
        max_turns=20,
    )


@pytest.mark.parametrize(
    "variant", ["full", "no_tracker", "no_intra", "no_inter", "joint", "no_anticipate"]
)
@pytest.mark.parametrize("level", ["full", "summary"])
@pytest.mark.parametrize("fail", [False, True])
async def test_v2_native_events_survive_real_pipeline_and_private_boundary(
    tmp_path, variant, level, fail
):
    audit = AuditLogger("gp-audit-fixture", output_dir=tmp_path, run_id="gp", level=level)
    client = HELPERS.Client({"generator": ["invalid"] * 4} if fail else {}, ask=True)
    session = HELPERS.session(variant, client=client)
    published = []

    async def ui_sink(item):
        published.append(item)

    result = await run_interaction(
        episode=episode(audit),
        assistant=session,
        audit=audit,
        update_memory=False,
        event_sink=ui_sink,
    )
    assert result.status == ("failed" if fail else "completed")
    saved = events(audit)
    kinds = [e["event_type"] for e in saved]
    terminal = "assistant_generation_failed" if fail else "assistant_generation_completed"
    metadata = saved[kinds.index(terminal)]["payload"]["llm_call"]
    gp = metadata["goal_progression"]
    assert gp["architecture_version"] == "v2_contracts" and gp["committed"] is (not fail)
    for role in session.settings.roles:
        requested = kinds.index("assistant_" + role + "_requested")
        raw = kinds.index("assistant_" + role + "_raw_result")
        assert requested < raw < kinds.index(terminal)
        recorded = saved[raw]["payload"]
        assert recorded["raw_output"] and recorded["schema_version"] == 2
        assert "input_hash" in recorded and "wall_seconds" in recorded
        assert recorded["status"] == "received"
        if not (fail and role == "generator"):
            assert raw < kinds.index("assistant_" + role + "_validated")
    if fail:
        assert kinds.count("assistant_generator_raw_result") == 4
        assert "assistant_gp_turn_failed" in kinds
        assert "assistant_gp_state_committed" not in kinds
        assert metadata["visible_response_tokens"] == 0 and not session.interaction_state
    else:
        committed = saved[kinds.index("assistant_gp_state_committed")]["payload"]
        reply = saved[kinds.index(terminal)]["payload"]["assistant_message"]
        assert committed["final_reply"] == reply == session.history[-1].content
        assert committed["receipt"]["issued_requests"]
        assert metadata["visible_response_tokens"] == len(reply.split())
        restored = HELPERS.session(variant)
        restored.restore_events(saved)
        assert restored.export_checkpoint() == session.export_checkpoint()
        assert restored.client.calls == []
        assert kinds.index("assistant_gp_contract_compiled") < kinds.index(
            "assistant_gp_reply_composed"
        )
        assert kinds.index("assistant_gp_reply_composed") < kinds.index(
            "assistant_gp_state_committed"
        )
    raw_events = json.dumps(saved)
    assert "SECRET_REASONING" not in raw_events
    for call in client.calls:
        model_input = json.dumps(call["context"])
        for forbidden in ("PRIVATE_INTENT", "SECRET_REASONING", "satisfaction", '"N1"'):
            assert forbidden not in model_input
    for channel in (json.dumps(published), (audit.run_dir / "transcript.jsonl").read_text()):
        for forbidden in ("assistant_gp_", "PRIVATE_INTENT", "SECRET_REASONING", "remaining_work"):
            assert forbidden not in channel


async def test_role_output_is_on_disk_before_next_stage_starts(tmp_path):
    audit = AuditLogger("gp-audit-fixture", output_dir=tmp_path, run_id="immediate")

    class CheckingClient(HELPERS.Client):
        async def generate(self, **kwargs):
            role = json.loads(kwargs["messages"][1]["content"])["role"]
            kinds = [e["event_type"] for e in events(audit)]
            if role in {"intra", "inter"}:
                assert "assistant_tracker_raw_result" in kinds
                assert "assistant_tracker_validated" in kinds
            if role == "generator":
                for owner in ("intra", "inter"):
                    assert "assistant_" + owner + "_raw_result" in kinds
                    assert "assistant_" + owner + "_validated" in kinds
                assert "assistant_gp_contract_compiled" in kinds
            return await super().generate(**kwargs)

    session = HELPERS.session(client=CheckingClient())
    await session.respond("Make a card.", audit_sink=lambda k, p: audit.log(k, 1, p))
    assert events(audit)[-1]["event_type"] == "assistant_gp_state_committed"


async def test_required_tracker_failure_is_saved_without_generator_fallback(tmp_path):
    audit = AuditLogger("gp-audit-fixture", output_dir=tmp_path, run_id="failure")
    client = HELPERS.Client({"tracker": ["INVALID_TRACKER"] * 4})
    session = HELPERS.session(client=client)
    with pytest.raises(RecoveryExhausted):
        await session.respond("Make a card.", audit_sink=lambda k, p: audit.log(k, 1, p))
    kinds = [e["event_type"] for e in events(audit)]
    assert kinds.count("assistant_tracker_raw_result") == 4
    assert kinds.count("assistant_tracker_validation_failed") == 4
    assert "assistant_generator_requested" not in kinds
    assert kinds[-1] == "assistant_gp_turn_failed"


async def test_write_ahead_commit_without_pipeline_ack_is_not_replayed(tmp_path):
    audit = AuditLogger("gp-audit-fixture", output_dir=tmp_path, run_id="no-ack")
    session = HELPERS.session()
    await session.respond("Make a card.", audit_sink=lambda k, p: audit.log(k, 1, p))
    with pytest.raises(ValueError, match="no acknowledged"):
        HELPERS.session().restore_events(events(audit))


@pytest.mark.parametrize("variant", HELPERS.VARIANTS)
def test_existing_batch_prepare_and_snapshot_support_all_six_profiles(variant):
    from interaction_pipeline.config import load_pipeline_config
    from interaction_pipeline.prepare import prepare_pipeline

    config = load_pipeline_config(
        overrides={
            "run": {
                "difficulty": "hard",
                "sample_retries": 0,
                "update_memory": False,
            }
        }
    )
    name = "qwen_3_6_27b_gp" + ("" if variant == "full" else "_" + variant)
    prepared = prepare_pipeline(
        config,
        assistant_model_profile=name,
        assistant_baseline="goal_progression",
        simulator_model_profile="deepseek_v4_flash_0731",
        max_turns=20,
    )
    snapshot = prepared.batch_config_snapshot(
        batch_id="synthetic-config-only",
        created_at="2026-09-10T00:00:00Z",
        samples=[],
        concurrency=1,
        update_memory=False,
    )
    gp = snapshot["assistant"]["goal_progression"]
    assert gp["architecture_version"] == "v2_contracts" and gp["variant"] == variant
    assert gp["max_in_flight_requests"] == 32
    assert gp["context"]["hard_context_tokens"] == 262144
    assert gp["realization"]["output_format"] == "json_units"
    assert set(gp["prompts"]) == set(HELPERS.profile(variant).goal_progression.roles)
