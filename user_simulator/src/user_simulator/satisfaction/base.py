from typing import Protocol

from user_simulator.domain.dag import DagNode
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import SatisfactionUpdateResult
from user_simulator.domain.state import EpisodeState


class SatisfactionUpdater(Protocol):
    async def update(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        exposed_nodes: list[DagNode],
    ) -> SatisfactionUpdateResult: ...
