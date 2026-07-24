from user_simulator.domain.dag import Sample
from user_simulator.graph.navigator import GraphNavigator


def test_outgoing_is_numeric_and_end_separate() -> None:
    sample = Sample.model_validate(
        {
            "sample_id": "s",
            "reason_dag": {
                "nodes": [
                    {"node_id": f"N{i}", "node_type": "intent", "node_intent": str(i)}
                    for i in range(1, 11)
                ]
                + [{"node_id": "END", "node_type": "terminal", "node_intent": "done"}],
                "edges": [
                    {"edge_id": "a", "source": "N1", "target": "N10"},
                    {"edge_id": "b", "source": "N1", "target": "N2"},
                    {"edge_id": "c", "source": "N1", "target": "END"},
                ],
            },
        }
    )
    navigator = GraphNavigator(sample)
    assert navigator.outgoing("N1") == (["N2", "N10"], True)
    assert navigator.next_backbone("N1") == "N2"
