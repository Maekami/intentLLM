import random

from user_simulator.domain.enums import Difficulty, RealizationMode
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy


def test_selection_by_difficulty() -> None:
    queue = ["N1", "N2", "N3"]
    policy = DifficultySelectionPolicy()
    assert policy.select(queue, Difficulty.EASY, random.Random(1)).selected_nodes == queue
    assert policy.select(queue, Difficulty.HARD, random.Random(1)).selected_nodes == ["N1"]
    first = policy.select(queue, Difficulty.MEDIUM, random.Random(8)).selected_nodes
    second = policy.select(queue, Difficulty.MEDIUM, random.Random(8)).selected_nodes
    assert first == second
    assert first == queue[: len(first)]


def test_realization_by_difficulty_and_seed() -> None:
    policy = DifficultyRealizationPolicy(0.5)
    assert policy.choose_mode(Difficulty.EASY, random.Random()) == RealizationMode.CLEAR
    assert policy.choose_mode(Difficulty.HARD, random.Random()) == RealizationMode.ABSTRACT
    assert policy.choose_mode(Difficulty.MEDIUM, random.Random(2)) == RealizationMode.ABSTRACT
    assert policy.choose_mode(Difficulty.MEDIUM, random.Random(1)) == RealizationMode.CLEAR
