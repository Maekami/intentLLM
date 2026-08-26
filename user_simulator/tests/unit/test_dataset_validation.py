import json

import pytest

from user_simulator.data.loader import DatasetLoader
from user_simulator.data.validator import DatasetValidator
from user_simulator.domain.dag import DagEdge, Sample
from user_simulator.exceptions import DatasetValidationError


def _sample() -> Sample:
    return Sample.model_validate(
        {
            "sample_id": "fixture",
            "reason_dag": {
                "nodes": [
                    {"node_id": "N1", "node_type": "intent", "node_intent": "first"},
                    {"node_id": "N2", "node_type": "intent", "node_intent": "second"},
                    {"node_id": "END", "node_type": "terminal", "node_intent": "done"},
                ],
                "edges": [
                    {"edge_id": "E1", "source": "N1", "target": "N2"},
                    {"edge_id": "E2", "source": "N2", "target": "END"},
                ],
            },
        }
    )


def test_valid_sample() -> None:
    report = DatasetValidator().validate_sample(_sample())
    assert not report.errors


def test_missing_backbone_is_error() -> None:
    sample = _sample()
    sample.reason_dag.edges.pop()
    report = DatasetValidator().validate_sample(sample)
    assert any("backbone" in issue.message for issue in report.errors)


def test_nonconsecutive_outgoing_candidates_are_valid() -> None:
    sample = _sample()
    sample.reason_dag.nodes.insert(
        2, sample.reason_dag.nodes[1].model_copy(update={"node_id": "N3"})
    )
    sample.reason_dag.nodes.insert(
        3, sample.reason_dag.nodes[1].model_copy(update={"node_id": "N4"})
    )
    sample.reason_dag.edges = [
        sample.reason_dag.edges[0],
        DagEdge(edge_id="E2", source="N2", target="N3"),
        DagEdge(edge_id="E3", source="N3", target="N4"),
        DagEdge(edge_id="E4", source="N4", target="END"),
        DagEdge(edge_id="E5", source="N1", target="N4"),
    ]
    sample = Sample.model_validate(sample.model_dump())
    report = DatasetValidator().validate_sample(sample)
    assert not report.errors
    assert not report.warnings


def test_loader_reports_bad_json(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        DatasetLoader(path).load_all_samples()


def test_loader_methods(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(_sample().model_dump()) + "\n", encoding="utf-8")
    loader = DatasetLoader(path)
    assert loader.load_sample_by_id("fixture").sample_id == "fixture"
    assert loader.sample_randomly(5).sample_id == "fixture"
