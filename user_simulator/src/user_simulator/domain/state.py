from pydantic import BaseModel, ConfigDict, Field

from user_simulator.domain.enums import Difficulty, SatisfactionLevel
from user_simulator.domain.messages import ChatMessage


class EpisodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    difficulty: Difficulty
    current_frontier: str
    exposed_nodes: list[str]
    satisfaction: dict[str, SatisfactionLevel]
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    end_exposed: bool = False
    terminated: bool = False
    turn_index: int = 0
    random_seed: int
