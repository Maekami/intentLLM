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


def test_medium_selection_respects_inclusive_node_bounds() -> None:
    queue = ["N1", "N2", "N3", "N4", "N5"]
    policy = DifficultySelectionPolicy(medium_min_nodes=2, medium_max_nodes=4)

    for seed in range(20):
        selected = policy.select(queue, Difficulty.MEDIUM, random.Random(seed)).selected_nodes
        assert 2 <= len(selected) <= 4
        assert selected == queue[: len(selected)]


def test_equal_medium_bounds_select_fixed_prefix() -> None:
    queue = ["N1", "N2", "N3", "N4"]
    policy = DifficultySelectionPolicy(medium_min_nodes=3, medium_max_nodes=3)

    for seed in range(10):
        assert (
            policy.select(queue, Difficulty.MEDIUM, random.Random(seed)).selected_nodes == queue[:3]
        )


def test_medium_bounds_clamp_to_available_nodes() -> None:
    queue = ["N1", "N2"]
    policy = DifficultySelectionPolicy(medium_min_nodes=3, medium_max_nodes=5)

    assert policy.select(queue, Difficulty.MEDIUM, random.Random(1)).selected_nodes == queue


def test_realization_by_difficulty_and_seed() -> None:
    policy = DifficultyRealizationPolicy(0.5)
    assert policy.choose_mode(Difficulty.EASY, random.Random()) == RealizationMode.CLEAR
    assert policy.choose_mode(Difficulty.HARD, random.Random()) == RealizationMode.ABSTRACT
    assert policy.choose_mode(Difficulty.MEDIUM, random.Random(2)) == RealizationMode.ABSTRACT
    assert policy.choose_mode(Difficulty.MEDIUM, random.Random(1)) == RealizationMode.CLEAR
