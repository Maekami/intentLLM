from user_simulator.config import GenerationSettings
from user_simulator.controller.llm_controller import LLMController
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.llm.mock import MockStructuredLLMClient
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy
from user_simulator.realizer.llm_realizer import LLMUserRealizer
from user_simulator.satisfaction.llm_updater import LLMSatisfactionUpdater


def fixture_sample() -> Sample:
    return Sample.model_validate(
        {
            "sample_id": "integration",
            "reason_dag": {
                "nodes": [
                    {"node_id": "N1", "node_type": "intent", "node_intent": "need a plan"},
                    {"node_id": "N2", "node_type": "intent", "node_intent": "has a constraint"},
                    {"node_id": "N3", "node_type": "intent", "node_intent": "wants final output"},
                    {"node_id": "END", "node_type": "terminal", "node_intent": "done"},
                ],
                "edges": [
                    {"edge_id": "E1", "source": "N1", "target": "N2"},
                    {"edge_id": "E2", "source": "N2", "target": "N3"},
                    {"edge_id": "E3", "source": "N3", "target": "END"},
                    {"edge_id": "E4", "source": "N1", "target": "N3"},
                    {"edge_id": "E5", "source": "N2", "target": "END"},
                ],
            },
            "retained_original_fields": {"task_summary": "plan", "task_expectation": "done"},
        }
    )


def components(responses):
    client = MockStructuredLLMClient(responses)
    generation = GenerationSettings(temperature=0, max_completion_tokens=100)
    return (
        client,
        LLMController(client, generation),
        LLMSatisfactionUpdater(client, generation),
        LLMUserRealizer(client, generation, generation),
    )


async def test_complete_episode_exposes_multiple_and_terminates(tmp_path) -> None:
    responses = [
        {
            "user_message": "Can you help with a plan?",
            "selected_node_ids": ["N1"],
            "realization_mode": "clear",
            "coverage": [{"node_id": "N1", "covered": True}],
            "contains_unsupported_intent": False,
            "summary": "initial",
        },
        {
            "decisions": [
                {"node_id": "N2", "exposable": True, "reason": "elicited"},
                {"node_id": "N3", "exposable": True, "reason": "elicited"},
            ],
            "end_reachable": False,
            "summary": "expose both",
        },
        {
            "updates": [
                {
                    "node_id": "N1",
                    "status": "partially_satisfied",
                    "reason": "The assistant offered a useful outline but omitted constraints.",
                },
                {
                    "node_id": "N2",
                    "status": "unsatisfied",
                    "reason": "The assistant has not addressed this newly exposed constraint.",
                },
                {
                    "node_id": "N3",
                    "status": "unsatisfied",
                    "reason": "The assistant has not supplied the requested final output.",
                },
            ],
            "summary": "not done",
        },
        {
            "user_message": "Please finish the plan with my constraint and final output.",
            "selected_node_ids": ["N1", "N2", "N3"],
            "realization_mode": "clear",
            "coverage": [
                {"node_id": "N1", "covered": True},
                {"node_id": "N2", "covered": True},
                {"node_id": "N3", "covered": True},
            ],
            "contains_unsupported_intent": False,
            "summary": "followup",
        },
        {
            "decisions": [],
            "end_reachable": True,
            "summary": "end structurally reachable",
        },
        {
            "updates": [
                {
                    "node_id": "N1",
                    "status": "satisfied",
                    "reason": "The assistant supplied the complete requested plan.",
                },
                {
                    "node_id": "N2",
                    "status": "satisfied",
                    "reason": "The assistant incorporated the important stated constraint.",
                },
                {
                    "node_id": "N3",
                    "status": "satisfied",
                    "reason": "The assistant delivered the requested final output.",
                },
            ],
            "summary": "all done",
        },
    ]
    client, controller, updater, realizer = components(responses)
    audit = AuditLogger(
        "integration", output_dir=tmp_path, run_id="test-run", config_snapshot={"x": 1}
    )
    episode = Episode(
        sample=fixture_sample(),
        difficulty=Difficulty.EASY,
        seed=4,
        controller=controller,
        satisfaction_updater=updater,
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=realizer,
        audit_logger=audit,
    )
    initial = await episode.start()
    assert initial.user_message
    assert "INITIAL TURN" in client.calls[0]["messages"][-1]["content"]
    followup = await episode.submit_assistant("Tell me the relevant details.")
    assert episode.state.exposed_nodes == ["N1", "N2", "N3"]
    assert not followup.terminal
    terminal = await episode.submit_assistant("Here is the complete result.")
    assert terminal.terminal
    assert episode.state.terminated
    assert "END" not in episode.state.exposed_nodes
    assert "END" not in episode.state.satisfaction
    assert len(client.calls) == 6
    event_types = [event.event_type for event in audit.events]
    assert "controller_prefix_normalized" in event_types
    assert "satisfaction_applied" in event_types
    assert "episode_terminated" in event_types
    assert (audit.run_dir / "events.jsonl").exists()
    assert (audit.run_dir / "transcript.jsonl").exists()
    assert (audit.run_dir / "final_state.json").exists()


async def test_natural_backbone_exposure() -> None:
    responses = [
        {
            "user_message": "Help.",
            "selected_node_ids": ["N1"],
            "realization_mode": "clear",
            "coverage": [{"node_id": "N1", "covered": True}],
            "contains_unsupported_intent": False,
            "summary": "initial",
        },
        {
            "decisions": [
                {"node_id": "N2", "exposable": False, "reason": "not elicited"},
                {"node_id": "N3", "exposable": False, "reason": "prefix"},
            ],
            "end_reachable": False,
            "summary": "none",
        },
        {
            "updates": [
                {
                    "node_id": "N1",
                    "status": "satisfied",
                    "reason": "The assistant fully addressed the first exposed need.",
                }
            ],
            "summary": "done",
        },
        {
            "user_message": "There is another constraint.",
            "selected_node_ids": ["N2"],
            "realization_mode": "clear",
            "coverage": [{"node_id": "N2", "covered": True}],
            "contains_unsupported_intent": False,
            "summary": "next",
        },
    ]
    _, controller, updater, realizer = components(responses)
    episode = Episode(
        sample=fixture_sample(),
        difficulty=Difficulty.EASY,
        seed=1,
        controller=controller,
        satisfaction_updater=updater,
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=realizer,
    )
    await episode.start()
    result = await episode.submit_assistant("Addressing only the first need.")
    assert result.user_message == "There is another constraint."
    assert episode.state.current_frontier == "N2"
    assert episode.state.exposed_nodes == ["N1", "N2"]


from user_simulator.audit.logger import AuditLogger
