import pytest

from user_simulator.audit.logger import AuditLogger
from user_simulator.config import EnvironmentSettings, load_config
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty, SatisfactionLevel
from user_simulator.domain.results import (
    NodeSatisfactionDecision,
    SatisfactionUpdateResult,
)
from user_simulator.engine.episode import Episode
from user_simulator.exceptions import InvalidTerminalStateError
from user_simulator.factory import build_simulator_components
from user_simulator.mock_components import MockController, MockUserRealizer


def sample() -> Sample:
    return Sample.model_validate(
        {
            "sample_id": "mock-demo",
            "reason_dag": {
                "nodes": [
                    {
                        "node_id": "N1",
                        "node_type": "intent",
                        "node_intent": "first need",
                    },
                    {
                        "node_id": "N2",
                        "node_type": "intent",
                        "node_intent": "second need",
                    },
                    {
                        "node_id": "N3",
                        "node_type": "intent",
                        "node_intent": "third need",
                    },
                    {
                        "node_id": "END",
                        "node_type": "terminal",
                        "node_intent": "done",
                    },
                ],
                "edges": [
                    {"edge_id": "E1", "source": "N1", "target": "N2"},
                    {"edge_id": "E2", "source": "N2", "target": "N3"},
                    {"edge_id": "E3", "source": "N3", "target": "END"},
                    {"edge_id": "E4", "source": "N1", "target": "N3"},
                ],
            },
            "retained_original_fields": {"task_summary": "SECRET LATENT SUMMARY"},
        }
    )


class AllUnresolvedUntilEnd:
    async def update(
        self,
        *,
        history,
        latest_assistant_response,
        state,
        exposed_nodes,
    ):
        status = (
            SatisfactionLevel.SATISFIED if state.end_reachable else SatisfactionLevel.UNSATISFIED
        )
        return SatisfactionUpdateResult(
            updates=[
                NodeSatisfactionDecision(
                    node_id=node.node_id,
                    status=status,
                    reason=(
                        "The final mock answer supplies all requested assistance."
                        if state.end_reachable
                        else "The clarification turn supplies no assistant solution yet."
                    ),
                )
                for node in exposed_nodes
            ],
            summary="Deterministic multi-node selection fixture.",
        )


def mock_config():
    return load_config(
        cli_overrides={
            "components": {
                "controller": "mock_controller",
                "satisfaction_updater": "mock_satisfaction_updater",
                "user_realizer": "mock_user_realizer",
            }
        }
    )


@pytest.mark.parametrize(
    ("difficulty", "expected_selected", "expected_modes"),
    [
        (Difficulty.EASY, ["N1", "N2", "N3"], ["clear", "clear"]),
        (Difficulty.MEDIUM, ["N1"], ["abstract", "abstract"]),
        (Difficulty.HARD, ["N1"], ["abstract", "abstract"]),
    ],
)
async def test_complete_mock_episode_selection_sequences(
    difficulty, expected_selected, expected_modes
) -> None:
    config = mock_config()
    components = build_simulator_components(
        config=config, environment=EnvironmentSettings(openrouter_api_key="")
    )
    components.satisfaction_updater = AllUnresolvedUntilEnd()
    episode = Episode(
        sample=sample(),
        difficulty=difficulty,
        seed=42,
        controller=components.controller,
        satisfaction_updater=components.satisfaction_updater,
        selection_policy=components.selection_policy,
        realization_policy=components.realization_policy,
        user_realizer=components.user_realizer,
    )
    await episode.start()
    followup = await episode.submit_assistant("Which details should I use?")
    assert not followup.terminal
    terminal = await episode.submit_assistant("Here is the complete answer.")
    assert terminal.terminal
    realizer = components.user_realizer
    assert isinstance(realizer, MockUserRealizer)
    assert realizer.calls[1]["selected_node_ids"] == expected_selected
    assert [call["mode"] for call in realizer.calls] == expected_modes


async def test_mock_demo_episode_starts_without_api_key() -> None:
    config = mock_config()
    components = build_simulator_components(
        config=config, environment=EnvironmentSettings(openrouter_api_key="")
    )
    episode = Episode(
        sample=sample(),
        difficulty=Difficulty.EASY,
        seed=1,
        controller=components.controller,
        satisfaction_updater=components.satisfaction_updater,
        selection_policy=components.selection_policy,
        realization_policy=components.realization_policy,
        user_realizer=components.user_realizer,
        audit_logger=AuditLogger("mock-demo", enabled=False),
    )
    initial = await episode.start()
    assert initial.user_message


class EndInconsistentController(MockController):
    async def decide(self, **kwargs):
        result = await super().decide(**kwargs)
        return result.model_copy(update={"end_reachable": False})


class AlwaysSatisfied:
    async def update(self, *, exposed_nodes, **kwargs):
        return SatisfactionUpdateResult(
            updates=[
                NodeSatisfactionDecision(
                    node_id=node.node_id,
                    status=SatisfactionLevel.SATISFIED,
                    reason="The assistant supplied complete help for this final need.",
                )
                for node in exposed_nodes
            ],
            summary="All exposed needs have assistant evidence.",
        )


async def test_final_node_deadlock_raises_clear_error() -> None:
    one_node = Sample.model_validate(
        {
            "sample_id": "deadlock",
            "reason_dag": {
                "nodes": [
                    {
                        "node_id": "N1",
                        "node_type": "intent",
                        "node_intent": "one need",
                    },
                    {
                        "node_id": "END",
                        "node_type": "terminal",
                        "node_intent": "done",
                    },
                ],
                "edges": [{"edge_id": "E1", "source": "N1", "target": "END"}],
            },
        }
    )
    config = mock_config()
    components = build_simulator_components(config=config)
    episode = Episode(
        sample=one_node,
        difficulty=Difficulty.EASY,
        seed=1,
        controller=EndInconsistentController(),
        satisfaction_updater=AlwaysSatisfied(),
        selection_policy=components.selection_policy,
        realization_policy=components.realization_policy,
        user_realizer=components.user_realizer,
    )
    await episode.start()
    with pytest.raises(InvalidTerminalStateError):
        await episode.submit_assistant("Complete answer.")
