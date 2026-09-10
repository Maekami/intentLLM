from __future__ import annotations

import json
from dataclasses import replace

import pytest
from conftest import make_trace, turn_events

from intent_metrics.aitr import (
    AITRJudgeResponse,
    AITRJudgment,
    OpenRouterAITRJudge,
    score_to_aitr,
)
from intent_metrics.cache import AITRCache
from intent_metrics.config import (
    AITREnvironmentSettings,
    load_aitr_config,
)
from intent_metrics.evaluator import aggregate_metrics, evaluate_trace
from intent_metrics.prompt import build_aitr_messages
from intent_metrics.report import dumps_json_two_decimals


class FakeJudge:
    model_id = "openai/gpt-5.6-luna"
    reasoning_effort = "high"
    config_hash = "f" * 64
    prompt_version = "v1"
    prompt_hash = "e" * 64

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, messages):
        self.calls += 1
        judgment = AITRJudgment(reason="Focused progress and one useful question.", score=2.6)
        return AITRJudgeResponse(
            judgment=judgment,
            raw_response=judgment.model_dump(mode="json"),
            metadata={"request_id": "fixture"},
        )


class FailingJudge(FakeJudge):
    async def judge(self, messages):
        self.calls += 1
        raise RuntimeError("judge unavailable")


class CapturingStructuredClient:
    def __init__(self) -> None:
        self.calls = []
        self.last_call_metadata = {"request_id": "structured-fixture", "messages": "hidden"}

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return AITRJudgment(reason="Appropriately direct.", score=3.0)


def complete_trace(tmp_path):
    events = turn_events(
        1,
        after={"N1": "satisfied", "N2": "satisfied"},
        newly_exposed=["N2"],
        output_tokens=400,
    )
    return make_trace(events=events, turns=1, run_dir=tmp_path / "run")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1.0, 0.0), (1.5, 25.0), (2.0, 50.0), (2.5, 75.0), (3.0, 100.0)],
)
def test_aitr_score_conversion(raw, expected) -> None:
    normalized, aitr = score_to_aitr(raw)
    assert normalized == expected / 100
    assert aitr == expected


def test_aitr_validation_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError):
        AITRJudgment(reason="invalid", score=3.1)


async def test_openrouter_judge_reuses_strict_structured_client_contract(tmp_path) -> None:
    client = CapturingStructuredClient()
    config = load_aitr_config()
    environment = AITREnvironmentSettings(
        openrouter_api_key="",
    )
    judge = OpenRouterAITRJudge(config, environment, client=client)
    result = await judge.judge(complete_trace(tmp_path).conversation)
    request = client.calls[0]
    assert request["response_model"] is AITRJudgment
    assert request["schema_name"] == "aitr_judgment"
    assert request["prompt_metadata"]["prompt_version"] == "v1"
    assert len(request["prompt_metadata"]["prompt_hash"]) == 64
    assert request["prompt_metadata"]["prompt_path"].endswith("metrics/configs/prompts/aitr.yaml")
    assert request["messages"][1]["content"].startswith("<conversation>")
    assert result.judgment.score == 3.0
    assert "messages" not in result.metadata


async def test_aitr_cache_is_reused_only_for_matching_configuration(tmp_path) -> None:
    trace = complete_trace(tmp_path)
    judge = FakeJudge()
    cache = AITRCache(tmp_path / "cache.json")
    first = await evaluate_trace(trace, judge=judge, cache=cache)
    second = await evaluate_trace(trace, judge=judge, cache=cache)
    assert judge.calls == 1
    assert first.aitr == pytest.approx(80.0)
    assert not first.aitr_cached
    assert second.aitr == pytest.approx(80.0)
    assert second.aitr_cached
    saved = json.loads((tmp_path / "cache.json").read_text())
    entry = next(iter(saved["entries"].values()))
    assert entry["conversation_hash"]
    assert entry["judge_model"] == judge.model_id
    assert entry["reasoning_effort"] == judge.reasoning_effort
    assert entry["judge_config_hash"] == judge.config_hash
    assert entry["prompt_version"] == "v1"
    assert entry["prompt_hash"] == judge.prompt_hash
    assert entry["raw_judge_response"]["score"] == 2.6

    changed_prompt_judge = FakeJudge()
    changed_prompt_judge.prompt_hash = "d" * 64
    changed = await evaluate_trace(trace, judge=changed_prompt_judge, cache=cache)
    assert changed_prompt_judge.calls == 1
    assert not changed.aitr_cached


async def test_failed_aitr_is_explicit_and_not_dropped_from_aggregate(tmp_path) -> None:
    record = await evaluate_trace(complete_trace(tmp_path), judge=FailingJudge())
    report = aggregate_metrics([record])
    hard = report["models"]["fake/model"]["hard"]
    assert record.status == "aitr_error"
    assert "judge unavailable" in record.aitr_error
    assert hard["aitr"] is None
    assert hard["diagnostics"]["aitr_api_failures"] == 1


def test_aitr_prompt_receives_only_visible_conversation() -> None:
    trace = make_trace(
        events=turn_events(1, after={"N1": "satisfied", "N2": "satisfied"}),
        turns=1,
    )
    rendered = build_aitr_messages(trace.conversation)[1]["content"]
    assert "Please help me." in rendered
    assert "Answer 1" in rendered
    assert "fake/model" not in rendered
    assert "N1" not in rendered
    assert "hard" not in rendered


async def test_macro_average_and_two_decimal_json(tmp_path) -> None:
    first = await evaluate_trace(complete_trace(tmp_path), judge=FakeJudge())
    second = replace(
        first,
        episode_id="fixture:second",
        all_node_exposure_turn=3,
        all_node_satisfaction_turn=5,
        assistant_tokens=800,
        aitr=60.0,
        aitr_raw_score=2.2,
        aitr_normalized_score=0.6,
    )
    report = aggregate_metrics([first, second])
    hard = report["models"]["fake/model"]["hard"]
    assert hard["avg_all_node_exposure_turns"] == 2.0
    assert hard["avg_all_node_satisfaction_turns"] == 3.0
    assert hard["avg_tokens"] == 600.0
    assert hard["aitr"] == 70.0
    serialized = dumps_json_two_decimals(report)
    assert '"avg_tokens": 600.00' in serialized
    assert '"aitr": 70.00' in serialized
    assert json.loads(serialized)["models"]["fake/model"]["hard"]["aitr"] == 70.0


@pytest.mark.parametrize("token_case", ["null", "invalid", "missing_usage", "missing_event"])
async def test_token_errors_are_metric_local_and_do_not_skip_aitr(tmp_path, token_case):
    trace = complete_trace(tmp_path)
    events = list(trace.events)
    completed = next(e for e in events if e["event_type"] == "assistant_generation_completed")
    if token_case == "missing_event":
        events.remove(completed)
    elif token_case == "missing_usage":
        completed["payload"].pop("llm_call")
    else:
        completed["payload"]["llm_call"]["visible_response_tokens"] = (
            None if token_case == "null" else -1
        )
    judge = FakeJudge()
    record = await evaluate_trace(replace(trace, events=tuple(events)), judge=judge)
    assert (record.all_node_exposure_turn, record.all_node_satisfaction_turn) == (1, 1)
    assert record.assistant_tokens is None
    assert record.token_error and record.status == "tokens_incomplete"
    assert any("E/S retained" in warning for warning in record.warnings)
    assert judge.calls == 1 and record.aitr == pytest.approx(80.0)
    aggregate = aggregate_metrics([record])
    hard = aggregate["models"]["fake/model"]["hard"]
    assert aggregate["evaluated_episode_count"] == 1
    assert aggregate["complete"] is False
    assert hard["status"] == "tokens_incomplete"
    assert hard["aitr"] == 80.0 and hard["avg_tokens"] is None


async def test_mixed_token_coverage_keeps_all_es_and_does_not_publish_subset_mean(tmp_path):
    known = await evaluate_trace(complete_trace(tmp_path), judge=FakeJudge())
    unknown = replace(
        known,
        episode_id="unknown",
        assistant_tokens=None,
        token_error="missing visible count",
        all_node_exposure_turn=3,
        all_node_satisfaction_turn=5,
    )
    aggregate = aggregate_metrics([known, unknown])
    hard = aggregate["models"]["fake/model"]["hard"]
    assert aggregate["evaluated_episode_count"] == aggregate["input_episode_count"] == 2
    assert hard["avg_all_node_exposure_turns"] == 2.0
    assert hard["avg_all_node_satisfaction_turns"] == 3.0
    assert hard["avg_tokens"] is None  # neither 400 (subset) nor 200 (zero imputation)
    assert hard["diagnostics"]["token_available_episode_count"] == 1
    assert hard["diagnostics"]["token_missing_episode_count"] == 1
    # Another model group remains independently complete; zero is a valid count.
    zero = replace(known, episode_id="zero", assistant_model="other/model", assistant_tokens=0)
    other = aggregate_metrics([unknown, zero])["models"]["other/model"]["hard"]
    assert other["avg_tokens"] == 0.0 and other["status"] == "complete"
    assert other["diagnostics"]["token_missing_episode_count"] == 0


async def test_one_missing_turn_never_publishes_partial_episode_tokens(tmp_path):
    events = []
    for turn in (1, 2):
        entries = turn_events(
            turn,
            after={"N1": "satisfied", "N2": "satisfied"},
            newly_exposed=["N2"] if turn == 1 else [],
            output_tokens=999,
        )
        entries[0]["payload"]["llm_call"]["visible_response_tokens"] = 7 if turn == 1 else None
        events.extend(entries)
    record = await evaluate_trace(make_trace(events=events, turns=2))
    assert record.assistant_tokens is None  # not the known first-turn count 7
    assert "turn 2" in record.token_error
    assert record.all_node_exposure_turn == record.all_node_satisfaction_turn == 1


async def test_token_error_and_aitr_failure_are_recorded_independently(tmp_path):
    trace = complete_trace(tmp_path)
    trace.events[0]["payload"]["llm_call"]["visible_response_tokens"] = None
    record = await evaluate_trace(trace, judge=FailingJudge())
    assert record.token_error and "judge unavailable" in record.aitr_error
    hard = aggregate_metrics([record])["models"]["fake/model"]["hard"]
    assert hard["status"] == "tokens_and_aitr_incomplete"
    assert hard["diagnostics"]["aitr_api_failures"] == 1
    assert hard["diagnostics"]["token_missing_episode_count"] == 1
    assert hard["avg_all_node_exposure_turns"] == hard["avg_all_node_satisfaction_turns"] == 1.0


async def test_token_error_preserves_aitr_cache_path(tmp_path):
    trace = complete_trace(tmp_path)
    trace.events[0]["payload"]["llm_call"]["visible_response_tokens"] = None
    judge = FakeJudge()
    cache = AITRCache(tmp_path / "cache.json")
    first = await evaluate_trace(trace, judge=judge, cache=cache)
    second = await evaluate_trace(trace, judge=judge, cache=cache)
    assert judge.calls == 1 and second.aitr_cached
    assert second.aitr == first.aitr
    assert second.token_error == first.token_error
