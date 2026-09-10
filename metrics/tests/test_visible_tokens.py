import json
from types import SimpleNamespace

import pytest

from intent_metrics.errors import MetricDataError
from intent_metrics.visible_tokens import recount_runs


class FixtureTokenizer:
    def encode(self, content, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=content.split())


def test_recount_includes_zero_delivery_failures_and_excludes_all_private_fields(tmp_path):
    one, failed = tmp_path / "one", tmp_path / "failed"
    one.mkdir()
    failed.mkdir()
    (one / "transcript.jsonl").write_text(
        "\n".join(
            json.dumps(message)
            for message in [
                {"role": "user", "content": "not counted"},
                {"role": "assistant", "content": "one two", "reasoning_content": "not counted"},
                {"role": "assistant", "content": "three four five"},
            ]
        )
    )
    (one / "events.jsonl").write_text("private calls and undelivered outputs are not read")
    (failed / "transcript.jsonl").write_text("")
    report = recount_runs([one, failed], FixtureTokenizer())
    assert report["episode_count"] == 2
    assert report["avg_visible_tokens"] == 2.5
    assert report["episodes"][0]["visible_tokens"] == 5
    assert report["episodes"][1]["visible_tokens"] == 0


def test_missing_transcript_cannot_silently_be_dropped(tmp_path):
    with pytest.raises(MetricDataError, match="missing visible transcript"):
        recount_runs([tmp_path], FixtureTokenizer())
