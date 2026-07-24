from user_simulator.data.validator import node_index
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.exceptions import GraphInvariantError


class GraphNavigator:
    def __init__(self, sample: Sample) -> None:
        self.sample = sample
        self._nodes = {node.node_id: node for node in sample.reason_dag.nodes}
        self._outgoing: dict[str, list[str]] = {node_id: [] for node_id in self._nodes}
        for edge in sample.reason_dag.edges:
            self._outgoing.setdefault(edge.source, []).append(edge.target)

    def node(self, node_id: str) -> DagNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise GraphInvariantError(f"unknown node: {node_id}") from exc

    def outgoing(self, node_id: str) -> tuple[list[str], bool]:
        targets = self._outgoing.get(node_id, [])
        intents = sorted((x for x in targets if x != "END"), key=node_index)
        return intents, "END" in targets

    def intent_nodes(self) -> list[str]:
        return sorted((x for x in self._nodes if x != "END"), key=node_index)

    def next_backbone(self, node_id: str) -> str | None:
        candidate = f"N{node_index(node_id) + 1}"
        return candidate if candidate in self._nodes else None
