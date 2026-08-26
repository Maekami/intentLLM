from typing import Protocol

from user_simulator.domain.dag import DagNode
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import ControllerResult
from user_simulator.domain.state import EpisodeState


class Controller(Protocol):
    async def decide(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        candidates: list[DagNode],
    ) -> ControllerResult: ...
