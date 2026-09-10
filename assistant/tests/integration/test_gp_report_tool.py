"""GP report bookkeeping must retain failures without counting internal output as visible."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["text", "structured"])
async def test_report_includes_undelivered_role_failures(tmp_path, monkeypatch, capsys, output_format):
    path = Path(__file__).resolve().parents[2] / "scripts/gp_analyze_iteration.py"
    spec = importlib.util.spec_from_file_location("gp_report_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    changes = module.changed_existing_fields
    assert changes({"action": "none"}, {"action": "none", "target": None}) == []
    raw = {"inter": {"action": "none", "target": "residual", "anchor_goal_ids": ["g"]}}
    original = json.dumps(raw)
    assert changes(raw, {"inter": {"action": "none", "target": None, "anchor_goal_ids": []}}) == [
        "$.inter.target", "$.inter.anchor_goal_ids",
    ]
    assert json.dumps(raw) == original
    assert changes({"items": [{"action": "advance"}]}, {"items": [{"action": "clarify"}]}) == ["$.items[0].action"]
    batch = tmp_path / "batch"
    batch.mkdir()
    runs = []
    for index, delivered in enumerate([True, False]):
        directory = batch / str(index)
        directory.mkdir()
        gp = {"fallbacks": [], "wall_seconds": 2, "calls": [
            {"role": "generator", "status": "received" if delivered else "failed", "attempt": 1},
        ]}
        if delivered:
            if output_format == "structured":
                gp["calls"][0].update(status="validated", raw_output=json.dumps({
                    "delivery_check": "PRIVATE_DRAFT", "reply": "Visible answer", "follow_up_question": None,
                }))
            gp["calls"][0]["usage"] = {"transport_attempts": [{"status": "success"}]}
            gp["intra_plan"] = {"items": [{"action": "clarify", "question": "Which option?",
                "question_status": "pending", "action_target": "Deliver supported work"}]}
        else:
            gp["calls"].insert(0, {"role": "intra", "status": "failed", "attempt": 1,
                "usage": {"transport_attempts": [
                    {"status": "failed", "finish_reason": "length", "raw_output": "PRIVATE_DRAFT"},
                    {"status": "failed", "finish_reason": "length", "raw_output": "PRIVATE_DRAFT"},
                ]}})
        event = {"event_type": "assistant_generation_completed" if delivered else "assistant_generation_failed",
                 "turn_index": 1, "payload": {"llm_call": {"goal_progression": gp}}}
        (directory / "events.jsonl").write_text(json.dumps(event) + "\n")
        runs.append({"sample_id": str(index), "status": "completed" if delivered else "failed",
                     "turns": int(delivered), "run_dir": str(directory),
                     "error": None if delivered else "scripted generation failure"})
    summary = {"runs": runs, "sample_count": 2, "completed": 1, "failed": 1, "total_retries": 0}
    (batch / "batch_summary.json").write_text(json.dumps(summary))
    config = {"dataset": {"sha256": "fixture"}, "assistant": {"goal_progression": {
        "format_retries": 0, "max_in_flight_requests": 1, "turn_timeout_seconds": 30,
        "prompts": {"generator": {"hash": "fixture", "output_format": output_format}},
    }}}
    (batch / "batch_config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(module, "TraceLoader", lambda: SimpleNamespace(load=lambda directory: directory))

    async def evaluate(directory):
        delivered = directory.name == "0"
        return SimpleNamespace(all_node_exposure_turn=1 if delivered else 21,
                               all_node_satisfaction_turn=1 if delivered else 21,
                               assistant_tokens=6 if delivered else 0)

    monkeypatch.setattr(module, "evaluate_trace", evaluate)
    await module.report(tmp_path, "fixture")
    output = capsys.readouterr().out
    assert "All-node E: **11**; all-node S: **11**" in output
    assert "Episode-level visible tokens: **3**" in output
    assert f"Generator output format: `{output_format}`" in output
    assert "Delivered turns: 1; clean turns: 1" in output
    assert "Failed generation turns: 1" in output
    assert "| generator | failed | 1 |" in output
    status = "validated" if output_format == "structured" else "received"
    assert f"| generator | {status} | 1 |" in output
    assert "scripted generation failure" in output
    assert "| intra | failed | length | 2 |" in output
    assert "| generator | success | unrecorded | 1 |" in output
    assert "without transport-attempt records: `{'generator': 1}`" in output
    assert "question-status labels: `{'pending': 1}`" in output
    assert "mean=3, max=3" in output
    assert "PRIVATE_DRAFT" not in output
    assert "**False**" in output
