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
