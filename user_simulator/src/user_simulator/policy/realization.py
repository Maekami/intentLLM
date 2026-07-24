import random
from typing import Protocol

from user_simulator.domain.enums import Difficulty, RealizationMode


class RealizationPolicy(Protocol):
    def choose_mode(self, difficulty: Difficulty, rng: random.Random) -> RealizationMode: ...


class DifficultyRealizationPolicy:
    def __init__(self, medium_clear_probability: float = 0.5) -> None:
        if not 0 <= medium_clear_probability <= 1:
            raise ValueError("medium clear probability must be between 0 and 1")
        self.medium_clear_probability = medium_clear_probability

    def choose_mode(self, difficulty: Difficulty, rng: random.Random) -> RealizationMode:
        if difficulty == Difficulty.EASY:
            return RealizationMode.CLEAR
        if difficulty == Difficulty.HARD:
            return RealizationMode.ABSTRACT
        return (
            RealizationMode.CLEAR
            if rng.random() < self.medium_clear_probability
            else RealizationMode.ABSTRACT
        )
