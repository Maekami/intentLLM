from user_simulator.domain.state import EpisodeState
from user_simulator.engine.transitions import all_exposed_satisfied


def should_terminate(state: EpisodeState) -> bool:
    return state.end_exposed and all_exposed_satisfied(state.exposed_nodes, state.satisfaction)
