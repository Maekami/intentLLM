"""Read-only v2 reporting uses committed visible receipts, not write-ahead records."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import yaml


async def test_v2_report_understands_new_fields_and_deduplicates_audit_retry(
    tmp_path, monkeypatch, capsys
):
    path = Path(__file__).resolve().parents[2] / "scripts/gp_analyze_iteration.py"
    spec = importlib.util.spec_from_file_location("gp_v2_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = tmp_path / "batch"
    run = batch / "sample"
    run.mkdir(parents=True)
    structured = {"units": [{"unit_id": "intra:g0001:0", "text": "VISIBLE"}], "issues": []}
    gp = {
        "architecture_version": "v2_contracts",
        "wall_seconds": 1,
        "degradations": [],
        "committed_state": {"goals": []},
        "retry_counts": {"intra:output.parse": 1},
        "receipt": {
            "delivered_units": [{"unit_id": "intra:g0001:0"}],
            "issued_requests": [{"need_id": "n0001"}],
        },
        "calls": [
            {
                "role": "generator",
                "attempt": 1,
                "status": "validated",
                "raw_output": json.dumps(structured),
                "structured_result": structured,
            }
        ],
    }
    native = {
        "event_type": "assistant_gp_contract_compiled",
        "turn_index": 1,
        "timestamp": "first",
        "payload": {
            "architecture_version": "v2_contracts",
            "episode_id": "e1",
            "event_id": "t1:a5",
            "contract": {"deliveries": [{}], "requests": [{}]},
        },
    }
    failed_commit = {
        "event_type": "assistant_gp_state_committed",
        "turn_index": 2,
        "payload": {
            "architecture_version": "v2_contracts",
            "episode_id": "e1",
            "event_id": "t2:a9",
            "receipt": {"delivered_units": [{}], "issued_requests": [{}]},
        },
    }
    recorded = [
        native,
        {**native, "timestamp": "retry"},
        {
            "event_type": "assistant_generation_completed",
            "turn_index": 1,
            "payload": {"llm_call": {"goal_progression": gp}},
        },
        failed_commit,
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in recorded) + "\n")
    (batch / "batch_summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "sample_id": "synthetic",
                        "status": "completed",
                        "turns": 1,
                        "run_dir": str(run),
                    }
                ],
                "sample_count": 1,
                "completed": 1,
                "failed": 0,
                "total_retries": 0,
            }
        )
    )
    (batch / "batch_config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {"sha256": "synthetic"},
                "assistant": {
                    "goal_progression": {
                        "architecture_version": "v2_contracts",
                        "max_in_flight_requests": 32,
                        "turn_timeout_seconds": 600,
                        "recovery": {"default_max_retries": 3},
                        "realization": {"output_format": "json_units"},
                        "prompts": {"generator": {"hash": "synthetic"}},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(module, "TraceLoader", lambda: SimpleNamespace(load=lambda p: p))

    async def evaluate(_):
        return SimpleNamespace(
            all_node_exposure_turn=2, all_node_satisfaction_turn=3, assistant_tokens=1
        )

    monkeypatch.setattr(module, "evaluate_trace", evaluate)
    assert len(module.read_events(run)) == 3
    await module.report(batch, "v2-synthetic")
    output = capsys.readouterr().out
    assert "Generator output format: `json_units`" in output
    assert "no baseline-win verdict" in output
    assert "Receipt facts: `{'delivered_units': 1, 'issued_requests': 1}`" in output
    assert "intra:output.parse" in output
