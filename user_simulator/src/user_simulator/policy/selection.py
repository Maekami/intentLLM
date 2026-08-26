import random
from typing import Protocol

from user_simulator.domain.enums import Difficulty
from user_simulator.domain.results import SelectionResult


class NodeSelectionPolicy(Protocol):
    def select(
        self, unresolved_queue: list[str], difficulty: Difficulty, rng: random.Random
    ) -> SelectionResult: ...


class DifficultySelectionPolicy:
    def __init__(
        self,
        medium_min_nodes: int = 1,
        medium_max_nodes: int | None = None,
    ) -> None:
        if medium_min_nodes < 1:
            raise ValueError("medium_min_nodes must be at least 1")
        if medium_max_nodes is not None and medium_max_nodes < medium_min_nodes:
            raise ValueError("medium_max_nodes must be greater than or equal to medium_min_nodes")
        self.medium_min_nodes = medium_min_nodes
        self.medium_max_nodes = medium_max_nodes

    def select(
        self, unresolved_queue: list[str], difficulty: Difficulty, rng: random.Random
    ) -> SelectionResult:
        if not unresolved_queue:
            raise ValueError("unresolved queue must be non-empty")
        if difficulty == Difficulty.EASY:
            selected = list(unresolved_queue)
        elif difficulty == Difficulty.MEDIUM:
            available = len(unresolved_queue)
            minimum = min(self.medium_min_nodes, available)
            maximum = min(self.medium_max_nodes or available, available)
            selected = unresolved_queue[: rng.randint(minimum, maximum)]
        else:
            selected = unresolved_queue[:1]
        return SelectionResult(unresolved_queue=list(unresolved_queue), selected_nodes=selected)
