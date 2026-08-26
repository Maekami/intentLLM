from user_simulator.audit.logger import read_git_commit
from user_simulator.controller.base import Controller
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.domain.enums import RealizationMode, SatisfactionLevel
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import (
    CandidateExposureDecision,
    ControllerResult,
    NodeCoverageDecision,
    SatisfactionUpdateResult,
    UserGenerationResult,
    make_node_satisfaction_decision,
)
from user_simulator.domain.state import EpisodeState
from user_simulator.realizer.base import UserRealizer
from user_simulator.satisfaction.base import SatisfactionUpdater


class MockController(Controller):
    """Deterministic controller used by the offline demo and CI."""

    def __init__(self) -> None:
        self.last_call_metadata = _mock_metadata()
        self.semantic_events: list[dict] = []

    async def decide(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        candidates: list[DagNode],
    ) -> ControllerResult:
        return ControllerResult(
            decisions=[
                CandidateExposureDecision(
                    node_id=node.node_id,
                    exposable=True,
                    reason="The deterministic mock advances this ordered candidate.",
                )
                for node in candidates
            ],
            summary="Deterministic mock exposure decision.",
        )


class MockSatisfactionUpdater(SatisfactionUpdater):
    """Keeps the active frontier unresolved until END has been exposed."""

    def __init__(self) -> None:
        self.last_call_metadata = _mock_metadata()
        self.semantic_events: list[dict] = []

    async def update(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        exposed_nodes: list[DagNode],
    ) -> SatisfactionUpdateResult:
        updates = []
        for node in exposed_nodes:
            status = (
                SatisfactionLevel.SATISFIED
                if state.end_exposed or node.node_id != state.current_frontier
                else state.satisfaction[node.node_id]
            )
            updates.append(
                make_node_satisfaction_decision(
                    node_id=node.node_id,
                    status=status,
                    reason=(
                        "The deterministic assistant turn supplies mock completion evidence."
                        if status == SatisfactionLevel.SATISFIED
                        else "The active frontier still needs a later mock assistant contribution."
                    ),
                    remaining_gap=None,
                )
            )
        return SatisfactionUpdateResult(
            updates=updates,
            summary="Deterministic mock satisfaction update.",
        )


class MockUserRealizer(UserRealizer):
    def __init__(self) -> None:
        self.last_call_metadata = _mock_metadata()
        self.semantic_events: list[dict] = []
        self.calls: list[dict] = []

    async def generate(
        self,
        *,
        selected_nodes: list[DagNode],
        unselected_unresolved_nodes: list[DagNode],
        satisfaction: dict[str, SatisfactionLevel],
        selected_remaining_gaps: dict[str, str],
        history: list[ChatMessage],
        latest_assistant_response: str | None,
        mode: RealizationMode,
        sample: Sample,
    ) -> UserGenerationResult:
        self.calls.append(
            {
                "selected_node_ids": [node.node_id for node in selected_nodes],
                "selected_remaining_gaps": dict(selected_remaining_gaps),
                "mode": mode.value,
                "initial": latest_assistant_response is None,
            }
        )
        fragments = []
        for node in selected_nodes:
            source = (
                node.surface_user_message
                if mode == RealizationMode.CLEAR and node.surface_user_message
                else node.reason_text or node.node_intent
            )
            fragments.append(source.strip().rstrip("."))
        if mode == RealizationMode.CLEAR:
            message = " Also, ".join(fragments) + "."
        else:
            message = "I'm still trying to work through this: " + "; ".join(fragments) + "."
        return UserGenerationResult(
            user_message=message,
            selected_node_ids=[node.node_id for node in selected_nodes],
            realization_mode=mode,
            coverage=[
                NodeCoverageDecision(node_id=node.node_id, covered=True) for node in selected_nodes
            ],
            contains_unsupported_task_content=False,
            summary="Deterministic mock user realization.",
        )


def _mock_metadata() -> dict:
    return {
        "model_id": "deterministic-mock",
        "model_profile": "mock",
        "git_commit": read_git_commit(),
        "prompt_name": None,
        "prompt_path": None,
        "prompt_hash": None,
        "schema_name": None,
        "schema_hash": None,
        "structured_validation_status": "valid",
        "semantic_validation_status": "valid",
        "transport_retry_count": 0,
        "semantic_retry_count": 0,
    }
