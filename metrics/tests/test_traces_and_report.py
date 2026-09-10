from __future__ import annotations

import json

import yaml

from intent_metrics.deterministic import FAILURE_TURN, compute_assistant_tokens
from intent_metrics.evaluator import evaluate_trace
from intent_metrics.models import EvaluationResult
from intent_metrics.report import write_evaluation_outputs
from intent_metrics.traces import TraceLoader, discover_run_dirs


def _write_fixture_run(tmp_path, *, budget_exhausted: bool = False):
    dataset = tmp_path / "DAG.jsonl"
    sample = {
        "sample_id": "sample-1",
        "reason_dag": {
            "nodes": [
                {"node_id": "N1", "node_type": "intent", "node_intent": "Get help."},
                {"node_id": "END", "node_type": "terminal", "node_intent": "Done."},
            ],
            "edges": [{"edge_id": "E1", "source": "N1", "target": "END"}],
        },
    }
    dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    run_dir = tmp_path / "sample-1_run"
    run_dir.mkdir()
    snapshot = {
        "pipeline": {"run": {"difficulty": "hard"}},
        "simulator": {"dataset_path": str(dataset)},
        "assistant_model_profile": {"model_id": "fake/model"},
        "sample_id": "sample-1",
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )

    events = [
        {
            "event_type": "episode_started",
            "sample_id": "sample-1",
            "turn_index": 0,
            "payload": {"difficulty": "hard", "initial_frontier": "N1"},
        }
    ]
    turns = 20 if budget_exhausted else 1
    transcript = [{"turn_index": 0, "role": "user", "content": "Help me."}]
    for turn in range(1, turns + 1):
        events.extend(
            [
                {
                    "event_type": "assistant_generation_completed",
                    "sample_id": "sample-1",
                    "turn_index": turn,
                    "payload": {"llm_call": {"model_id": "fake/model", "output_tokens": 10}},
                },
                {
                    "event_type": "assistant_message_received",
                    "sample_id": "sample-1",
                    "turn_index": turn,
                    "payload": {},
                },
                {
                    "event_type": "satisfaction_applied",
                    "sample_id": "sample-1",
                    "turn_index": turn,
                    "payload": {
                        "after": {"N1": "unsatisfied" if budget_exhausted else "satisfied"}
                    },
                },
            ]
        )
        transcript.append({"turn_index": turn, "role": "assistant", "content": f"Answer {turn}."})
        if turn < turns or budget_exhausted:
            transcript.append({"turn_index": turn, "role": "user", "content": f"More {turn}."})
    if budget_exhausted:
        # This models the current pipeline's generated-but-unsubmitted turn 21.
        events.append(
            {
                "event_type": "assistant_generation_completed",
                "sample_id": "sample-1",
                "turn_index": 21,
                "payload": {"llm_call": {"model_id": "fake/model", "output_tokens": 999}},
            }
        )
        events.append(
            {
                "event_type": "pipeline_run_failed",
                "sample_id": "sample-1",
                "turn_index": 20,
                "payload": {"error_type": "EpisodeTurnLimitError"},
            }
        )
    else:
        events.append(
            {
                "event_type": "pipeline_run_completed",
                "sample_id": "sample-1",
                "turn_index": 1,
                "payload": {"turns": 1},
            }
        )
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (run_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(message) + "\n" for message in transcript), encoding="utf-8"
    )
    (run_dir / "final_state.json").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "difficulty": "hard",
                "turn_index": turns,
                "terminated": not budget_exhausted,
            }
        ),
        encoding="utf-8",
    )
    return run_dir, dataset


def test_trace_loader_uses_snapshot_dataset_and_excludes_terminal_node(tmp_path) -> None:
    run_dir, _ = _write_fixture_run(tmp_path)
    trace = TraceLoader().load(run_dir)
    assert trace.sample_id == "sample-1"
    assert trace.difficulty == "hard"
    assert trace.assistant_model == "fake/model"
    assert trace.intent_node_ids == ("N1",)
    assert trace.initial_exposed_node_ids == ("N1",)


async def test_budget_exhaustion_is_valid_and_turn_twenty_one_tokens_are_excluded(
    tmp_path,
) -> None:
    run_dir, _ = _write_fixture_run(tmp_path, budget_exhausted=True)
    trace = TraceLoader().load(run_dir)
    assert trace.source_outcome == "budget_exhausted"
    assert trace.conversation[-1].role == "assistant"
    assert trace.conversation[-1].turn_index == 20
    assert compute_assistant_tokens(trace).assistant_tokens == 200
    record = await evaluate_trace(trace)
    assert record.all_node_exposure_turn == 0
    assert record.all_node_satisfaction_turn == FAILURE_TURN


def test_discover_run_dirs_uses_batch_summary_and_relocated_basename(tmp_path) -> None:
    run_dir, _ = _write_fixture_run(tmp_path)
    summary = {
        "runs": [
            {
                "sample_id": "sample-1",
                "run_dir": f"/stale/machine/path/{run_dir.name}",
            }
        ]
    }
    (tmp_path / "batch_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert discover_run_dirs(tmp_path) == [run_dir.resolve()]


async def test_report_writes_episode_and_two_decimal_aggregate_files(tmp_path) -> None:
    run_dir, _ = _write_fixture_run(tmp_path)
    record = await evaluate_trace(TraceLoader().load(run_dir))
    episode_path, aggregate_path, text_path = write_evaluation_outputs(
        tmp_path / "results", EvaluationResult(records=[record])
    )
    assert episode_path.is_file()
    assert aggregate_path.is_file()
    assert text_path.is_file()
    aggregate_text = aggregate_path.read_text()
    assert '"avg_all_node_exposure_turns": 0.00' in aggregate_text
    assert '"avg_all_node_satisfaction_turns": 1.00' in aggregate_text
    assert '"avg_tokens": 10.00' in aggregate_text
    assert '"aitr": null' in aggregate_text


async def test_unknown_visible_tokens_keep_episode_es_in_evaluation_and_reports(tmp_path):
    from intent_metrics.deterministic import compute_dag_turn_metrics
    from intent_metrics.evaluator import aggregate_metrics, evaluate_runs

    run_dir, dataset = _write_fixture_run(tmp_path)
    sample = json.loads(dataset.read_text())
    sample["reason_dag"]["nodes"].insert(
        1,
        {
            "node_id": "N2",
            "node_type": "intent",
            "node_intent": "Follow-up help.",
        },
    )
    sample["reason_dag"]["edges"] = [
        {"edge_id": "E1", "source": "N1", "target": "N2"},
        {"edge_id": "E2", "source": "N2", "target": "END"},
    ]
    dataset.write_text(json.dumps(sample) + "\n")
    events_path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events.insert(
        3,
        {
            "event_type": "nodes_exposed",
            "turn_index": 1,
            "payload": {"newly_exposed": ["N2"]},
        },
    )
    for event in events:
        if event["event_type"] == "assistant_generation_completed":
            event["payload"]["llm_call"]["visible_response_tokens"] = None
        if event["event_type"] == "satisfaction_applied":
            event["payload"]["after"]["N2"] = "satisfied"
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    loader = TraceLoader()
    dag = compute_dag_turn_metrics(loader.load(run_dir))
    assert (dag.all_node_exposure_turn, dag.all_node_satisfaction_turn) == (1, 1)
    result = await evaluate_runs([str(run_dir)], loader=loader)
    assert len(result.records) == 1
    assert result.failures == []
    record = result.records[0]
    assert record.assistant_tokens is None
    assert "visible tokens unknown" in record.token_error
    assert (record.all_node_exposure_turn, record.all_node_satisfaction_turn) == (1, 1)
    summary = aggregate_metrics(result.records, result.failures)
    assert summary["input_episode_count"] == summary["evaluated_episode_count"] == 1
    hard = summary["models"]["fake/model"]["hard"]
    assert hard["avg_all_node_exposure_turns"] == hard["avg_all_node_satisfaction_turns"] == 1.0
    assert hard["avg_tokens"] is None
    assert hard["diagnostics"]["token_missing_episode_count"] == 1
    episode_path, aggregate_path, text_path = write_evaluation_outputs(tmp_path / "results", result)
    saved_record = json.loads(episode_path.read_text())
    assert saved_record["assistant_tokens"] is None and saved_record["token_error"]
    assert json.loads(aggregate_path.read_text()) == summary
    assert "1/1 evaluated" in text_path.read_text()
    assert "token_missing_episodes=1" in text_path.read_text()


async def test_dag_corruption_still_excludes_episode_even_with_unknown_tokens(tmp_path):
    from intent_metrics.evaluator import evaluate_runs

    run_dir, _ = _write_fixture_run(tmp_path)
    path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event["event_type"] == "assistant_generation_completed":
            event["payload"]["llm_call"]["visible_response_tokens"] = None
        if event["event_type"] == "satisfaction_applied":
            event["payload"]["after"]["N1"] = "invalid satisfaction"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = await evaluate_runs([str(run_dir)], loader=TraceLoader())
    assert result.records == [] and len(result.failures) == 1
    assert "invalid status" in result.failures[0].message


def test_cli_preserves_es_outputs_when_token_counts_are_missing(tmp_path):
    from typer.testing import CliRunner

    from intent_metrics.cli import app

    run_dir, _ = _write_fixture_run(tmp_path)
    path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event["event_type"] == "assistant_generation_completed":
            event["payload"]["llm_call"]["visible_response_tokens"] = None
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    output_dir = tmp_path / "metrics-output"
    result = CliRunner().invoke(app, [str(run_dir), "--skip-aitr", "--output-dir", str(output_dir)])
    assert result.exit_code == 0, result.output
    report = json.loads((output_dir / "aggregate_metrics.json").read_text())
    assert report["evaluated_episode_count"] == report["input_episode_count"] == 1
    assert report["evaluation_errors"] == []
    assert report["models"]["fake/model"]["hard"]["avg_tokens"] is None
    assert "their E/S metrics are retained" in result.output
