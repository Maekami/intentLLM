"""GP audit aggregation selects outcomes without dropping hard or unreadable episodes."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def inspector(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "gp_iteration_inspect_test", scripts / "gp_iteration_inspect.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ({"status": "completed", "turns": 20, "outcome": "SUCCESS"}, True),
        ({"status": "failed", "outcome": "FAILURE_TURN_LIMIT", "turns": 20}, True),
        ({"status": "failed", "outcome": "INFRASTRUCTURE_FAILURE", "turns": 20}, False),
        ({"status": "failed", "error": "EpisodeTurnLimitError: max reached"}, True),
        ({"status": "failed", "error": "RecoveryExhausted: component failed"}, False),
        ({"status": "failed", "error": None, "turns": 20}, False),
    ],
)
def test_population_uses_outcome_not_number_of_turns(inspector, run, expected):
    assert inspector.completed_or_turn_limit(run) is expected


@pytest.mark.parametrize("population", ["all", "completed-or-turn-limit"])
@pytest.mark.parametrize("missing", [None, "metric", "tokens"])
async def test_aggregation_keeps_selected_failures_and_independent_token_availability(
    tmp_path, monkeypatch, inspector, population, missing
):
    from intent_metrics import evaluator, traces

    runs = [
        {
            "sample_id": name,
            "run_dir": str(tmp_path / name),
            "status": "completed" if name == "success" else "failed",
            "outcome": outcome,
            "turns": 20,
            "error": None if name == "success" else "synthetic failure",
        }
        for name, outcome in [
            ("success", "SUCCESS"),
            ("limit", "FAILURE_TURN_LIMIT"),
            ("infra", "INFRASTRUCTURE_FAILURE"),
        ]
    ]
    (tmp_path / "batch_config.yaml").write_text("{}")
    (tmp_path / "batch_summary.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "sample_count": 3,
                "completed": 1,
                "failed": 2,
                "infrastructure_failures": 1,
                "behavioral_failures": 1,
            }
        )
    )
    evaluated = []

    async def evaluate(directory):
        evaluated.append(directory.name)
        if missing == "metric" and directory.name == "limit":
            raise ValueError("Selected trace is unreadable")
        e, s = {"success": (2, 3), "limit": (5, 21), "infra": (1, 1)}[directory.name]
        return SimpleNamespace(
            all_node_exposure_turn=e,
            all_node_satisfaction_turn=s,
            assistant_tokens=None if missing == "tokens" else 10,
        )

    monkeypatch.setattr(traces, "TraceLoader", lambda: SimpleNamespace(load=lambda path: path))
    monkeypatch.setattr(evaluator, "evaluate_trace", evaluate)
    result = await inspector.inspect(tmp_path, metric_population=population)
    all_samples = population == "all"
    assert evaluated == (["success", "limit", "infra"] if all_samples else ["success", "limit"])
    assert result["metric_sample_count"] == (3 if all_samples else 2)
    assert result["metric_excluded_sample_count"] == (0 if all_samples else 1)
    assert result["metric_evaluation_failures"] == (1 if missing == "metric" else 0)
    assert len(result["samples"]) == 3
    assert result["samples"][2]["included_in_metrics"] is all_samples
    if not all_samples:
        assert "metric_error" not in result["samples"][2]
        assert result["samples"][2]["metric_exclusion_reason"]
    if missing == "metric":
        assert result["E"] is None and result["S"] is None
    else:
        assert result["E"] == pytest.approx(8 / 3 if all_samples else 3.5)
        assert result["S"] == pytest.approx(25 / 3 if all_samples else 12)
    assert result["visible_tokens"] == (10 if missing is None else None)
