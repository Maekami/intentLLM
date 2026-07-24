import random
from typing import Protocol

from user_simulator.domain.enums import Difficulty
from user_simulator.domain.results import SelectionResult


class NodeSelectionPolicy(Protocol):
    def select(
        self, unresolved_queue: list[str], difficulty: Difficulty, rng: random.Random
    ) -> SelectionResult: ...


class DifficultySelectionPolicy:
    def select(
        self, unresolved_queue: list[str], difficulty: Difficulty, rng: random.Random
    ) -> SelectionResult:
        if not unresolved_queue:
            raise ValueError("unresolved queue must be non-empty")
        if difficulty == Difficulty.EASY:
            selected = list(unresolved_queue)
        elif difficulty == Difficulty.MEDIUM:
            selected = unresolved_queue[: rng.randint(1, len(unresolved_queue))]
        else:
            selected = unresolved_queue[:1]
        return SelectionResult(unresolved_queue=list(unresolved_queue), selected_nodes=selected)
