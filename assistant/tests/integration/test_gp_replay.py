"""Local replay can resolve finished samples while a batch is still running."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gp_replay_test", Path(__file__).resolve().parents[2] / "scripts/replay_gp_turns.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def test_resolves_exact_sample_without_a_finished_batch_summary(tmp_path):
    sample = tmp_path / "synthetic_sample_0001"
    sample.mkdir()
    (sample / "events.jsonl").write_text("")
    (tmp_path / "synthetic_sample_extra_0002").mkdir()
    assert REPLAY.sample_directory(tmp_path, "synthetic_sample") == sample
    assert not (tmp_path / "batch_summary.json").exists()


def test_replay_refuses_missing_or_ambiguous_sample_without_summary(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        REPLAY.sample_directory(tmp_path, "synthetic")
    for name in ("synthetic_0001", "synthetic_0002"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "events.jsonl").write_text("")
    with pytest.raises(ValueError, match="found 2"):
        REPLAY.sample_directory(tmp_path, "synthetic")


def test_replay_prefers_existing_summary_mapping(tmp_path):
    (tmp_path / "batch_summary.json").write_text(
        json.dumps({"runs": [{"sample_id": "synthetic", "run_dir": "/old/location/resolved"}]})
    )
    assert REPLAY.sample_directory(tmp_path, "synthetic") == tmp_path / "resolved"
