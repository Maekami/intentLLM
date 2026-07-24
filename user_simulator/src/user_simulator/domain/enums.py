from enum import Enum


class SatisfactionLevel(str, Enum):
    UNSATISFIED = "unsatisfied"
    PARTIALLY_SATISFIED = "partially_satisfied"
    SATISFIED = "satisfied"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RealizationMode(str, Enum):
    CLEAR = "clear"
    ABSTRACT = "abstract"
