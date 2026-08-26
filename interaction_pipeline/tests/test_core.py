import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from assistant.baselines.base import BaseBaseline
from assistant.config import EnvironmentSettings as AssistantEnvironmentSettings
from assistant.config import GenerationSettings
from assistant.llm.base import GeneratedResponse
from assistant.session import AssistantSession
from user_simulator.audit.logger import AuditLogger
from user_simulator.config import EnvironmentSettings as SimulatorEnvironmentSettings
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.mock_components import MockController, MockSatisfactionUpdater, MockUserRealizer
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy

from interaction_pipeline.batch import run_batch
from interaction_pipeline.config import load_pipeline_config
from interaction_pipeline.core import run_interaction
from interaction_pipeline.prepare import BuiltInteraction, prepare_pipeline
from interaction_pipeline.web import _panel_html


class FakeAssistantClient:
    def __init__(self) -> None:
        self.profile = type("Profile", (), {"profile_name": "fake-assistant"})()
        self.last_call_metadata = {
            "model_id": "fake-assistant",
            "input_tokens": 1,
            "output_tokens": 9,
            "thinking_tokens": 4,
            "answer_tokens": 5,
            "reasoning_content": "hidden reasoning must not be audited",
        }

    async def generate(self, *, messages, generation):
        return GeneratedResponse(content="Here is the complete answer.")


class FinalizingAssistantSession(AssistantSession):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.finalizations = []

    def finalize_task(self, **values):
        self.finalizations.append(values)
        return {
            "stored": True,
            "task_id": values["task_id"],
            "path": "/tmp/fake-memory.json",
            "entry_count": 1,
            "reason": "added",
        }


def one_node_sample() -> Sample:
    return Sample.model_validate(
        {
            "sample_id": "pipeline-fixture",
            "reason_dag": {
                "nodes": [
                    {
                        "node_id": "N1",
                        "node_type": "intent",
                        "surface_user_message": "Please help me.",
                        "node_intent": "Get help.",
                    },
                    {"node_id": "END", "node_type": "terminal", "node_intent": "Done."},
                ],
                "edges": [{"edge_id": "E1", "source": "N1", "target": "END"}],
            },
        }
    )


async def test_complete_interaction_writes_jsonl_and_human_audit(tmp_path) -> None:
    sample = one_node_sample()
    audit = AuditLogger(sample.sample_id, output_dir=tmp_path, run_id="run", level="full")
    episode = Episode(
        sample=sample,
        difficulty=Difficulty.EASY,
        seed=1,
        controller=MockController(),
        satisfaction_updater=MockSatisfactionUpdater(),
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=MockUserRealizer(),
        audit_logger=audit,
    )
    assistant = AssistantSession(
        FakeAssistantClient(),
        GenerationSettings(max_completion_tokens=100),
        BaseBaseline(),
    )
    streamed = []

    async def sink(event):
        streamed.append(event)

    result = await run_interaction(
        episode=episode,
        assistant=assistant,
        audit=audit,
        event_sink=sink,
    )
    assert result.status == "completed"
    assert result.turns == 1
    assert [item["role"] for item in streamed if item["event_type"] == "message"] == [
        "user",
        "assistant",
    ]
    for name in (
        "events.jsonl",
        "events.txt",
        "transcript.jsonl",
        "transcript.txt",
        "config_snapshot.yaml",
        "final_state.json",
    ):
        assert (result.run_dir / name).exists()
    transcript = [
        json.loads(line) for line in (result.run_dir / "transcript.jsonl").read_text().splitlines()
    ]
    assert [item["role"] for item in transcript] == ["user", "assistant"]
    assert "SIMULATOR" not in (result.run_dir / "transcript.txt").read_text()
    assert "USER" in (result.run_dir / "transcript.txt").read_text()
    assert "ASSISTANT" in (result.run_dir / "transcript.txt").read_text()
    events = [
        json.loads(line) for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    assistant_event = next(
        item for item in events if item["event_type"] == "assistant_generation_completed"
    )
    llm_call = assistant_event["payload"]["llm_call"]
    assert llm_call["output_tokens"] == 9
    assert llm_call["thinking_tokens"] == 4
    assert llm_call["answer_tokens"] == 5
    assert "reasoning_content" not in llm_call


async def test_completed_interaction_finalizes_assistant_task_once(tmp_path) -> None:
    sample = one_node_sample()
    audit = AuditLogger(sample.sample_id, output_dir=tmp_path, run_id="memory", level="full")
    episode = Episode(
        sample=sample,
        difficulty=Difficulty.EASY,
        seed=1,
        controller=MockController(),
        satisfaction_updater=MockSatisfactionUpdater(),
        selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),
        user_realizer=MockUserRealizer(),
        audit_logger=audit,
    )
    assistant = FinalizingAssistantSession(
        FakeAssistantClient(),
        GenerationSettings(max_completion_tokens=100),
        BaseBaseline(),
    )

    result = await run_interaction(episode=episode, assistant=assistant, audit=audit)

    assert result.status == "completed"
    assert assistant.finalizations == [
        {
            "task_id": sample.sample_id,
            "success": True,
            "feedback": "Episode terminated normally after 1 assistant turns.",
        }
    ]
    events = [
        json.loads(line) for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(item["event_type"] == "assistant_memory_updated" for item in events)


class FakePreparedPipeline:
    def __init__(self) -> None:
        self.config = SimpleNamespace(run=SimpleNamespace(seed=10))

    def build(self, sample, *, output_dir, seed, run_id=None):
        audit = AuditLogger(
            sample.sample_id,
            output_dir=output_dir,
            run_id=run_id or f"{sample.sample_id}-{seed}",
            level="full",
        )
        episode = Episode(
            sample=sample,
            difficulty=Difficulty.EASY,
            seed=seed,
            controller=MockController(),
            satisfaction_updater=MockSatisfactionUpdater(),
            selection_policy=DifficultySelectionPolicy(),
            realization_policy=DifficultyRealizationPolicy(),
            user_realizer=MockUserRealizer(),
            audit_logger=audit,
        )
        assistant = AssistantSession(
            FakeAssistantClient(),
            GenerationSettings(max_completion_tokens=100),
            BaseBaseline(),
        )
        return BuiltInteraction(episode=episode, assistant=assistant, audit=audit)


async def test_batch_runs_multiple_samples_and_writes_summary(tmp_path) -> None:
    first = one_node_sample()
    second = first.model_copy(update={"sample_id": "pipeline-fixture-2"}, deep=True)
    progress_updates = []

    def progress(finished, total, result):
        progress_updates.append((finished, total, result.sample_id, result.status))

    batch_dir, results = await run_batch(
        FakePreparedPipeline(),
        [first, second],
        output_root=tmp_path,
        concurrency=2,
        progress_callback=progress,
    )
    assert [item.status for item in results] == ["completed", "completed"]
    assert [item[:2] for item in progress_updates] == [(1, 2), (2, 2)]
    assert {item[2] for item in progress_updates} == {
        "pipeline-fixture",
        "pipeline-fixture-2",
    }
    assert {item[3] for item in progress_updates} == {"completed"}
    summary = json.loads((batch_dir / "batch_summary.json").read_text())
    assert summary["sample_count"] == 2
    assert summary["completed"] == 2
    assert summary["sample_retries"] == 3
    assert summary["total_attempts"] == 2
    assert summary["total_retries"] == 0
    assert summary["retried_samples"] == 0
    assert [item["retry_count"] for item in summary["runs"]] == [0, 0]
    assert (batch_dir / "batch_summary.txt").exists()
    batch_config = yaml.safe_load((batch_dir / "batch_config.yaml").read_text())
    assert next(iter(batch_config)) == "run_overview"
    assert batch_config["batch"]["concurrency"] == 2
    assert batch_config["batch"]["sample_retries"] == 3
    assert batch_config["batch"]["samples"] == [
        {"sample_id": "pipeline-fixture", "seed": 10},
        {"sample_id": "pipeline-fixture-2", "seed": 11},
    ]


class PartiallyFailingPreparedPipeline(FakePreparedPipeline):
    def build(self, sample, *, output_dir, seed):
        if sample.sample_id.endswith("-2"):
            raise RuntimeError("fixture setup failure")
        return super().build(sample, output_dir=output_dir, seed=seed)


class FlakyPreparedPipeline(FakePreparedPipeline):
    def __init__(self, *, failures_before_success: int) -> None:
        super().__init__()
        self.failures_before_success = failures_before_success
        self.build_calls = 0
        self.seeds = []

    def build(self, sample, *, output_dir, seed, run_id=None):
        self.build_calls += 1
        self.seeds.append(seed)
        if self.build_calls <= self.failures_before_success:
            raise RuntimeError(f"transient fixture failure {self.build_calls}")
        return super().build(sample, output_dir=output_dir, seed=seed, run_id=run_id)


async def test_batch_isolates_setup_failure_and_keeps_jsonl_files(tmp_path) -> None:
    first = one_node_sample()
    second = first.model_copy(update={"sample_id": "pipeline-fixture-2"}, deep=True)
    progress_updates = []
    _, results = await run_batch(
        PartiallyFailingPreparedPipeline(),
        [first, second],
        output_root=tmp_path,
        concurrency=2,
        progress_callback=lambda finished, total, result: progress_updates.append(
            (finished, total, result.status)
        ),
    )
    assert [item.status for item in results] == ["completed", "failed"]
    assert [item[:2] for item in progress_updates] == [(1, 2), (2, 2)]
    assert {item[2] for item in progress_updates} == {"completed", "failed"}
    failed = results[1]
    assert "fixture setup failure" in failed.error
    for name in ("events.jsonl", "events.txt", "transcript.jsonl", "transcript.txt"):
        assert (failed.run_dir / name).exists()


async def test_batch_retries_entire_sample_three_times_by_default(tmp_path) -> None:
    prepared = FlakyPreparedPipeline(failures_before_success=3)
    progress_updates = []

    batch_dir, results = await run_batch(
        prepared,
        [one_node_sample()],
        output_root=tmp_path,
        concurrency=1,
        progress_callback=lambda finished, total, result: progress_updates.append(
            (finished, total, result.status)
        ),
    )

    assert [item.status for item in results] == ["completed"]
    assert prepared.build_calls == 4
    assert prepared.seeds == [10, 10, 10, 10]
    assert progress_updates == [(1, 1, "completed")]
    summary = json.loads((batch_dir / "batch_summary.json").read_text())
    assert summary["sample_retries"] == 3
    assert summary["total_attempts"] == 4
    assert summary["total_retries"] == 3
    assert summary["retried_samples"] == 1
    run = summary["runs"][0]
    assert run["retry_count"] == 3
    assert "attempt_count" not in run
    assert "attempt_history" not in run
    assert Path(run["run_dir"]).parent == batch_dir
    assert Path(run["run_dir"]).name == "pipeline-fixture_0001"
    assert [path for path in batch_dir.iterdir() if path.is_dir()] == [Path(run["run_dir"])]
    assert "transient fixture failure" not in (Path(run["run_dir"]) / "events.jsonl").read_text()


async def test_batch_stops_after_configured_retry_limit(tmp_path) -> None:
    prepared = FlakyPreparedPipeline(failures_before_success=3)

    batch_dir, results = await run_batch(
        prepared,
        [one_node_sample()],
        output_root=tmp_path,
        concurrency=1,
        sample_retries=2,
    )

    assert [item.status for item in results] == ["failed"]
    assert prepared.build_calls == 3
    summary = json.loads((batch_dir / "batch_summary.json").read_text())
    assert summary["sample_retries"] == 2
    assert summary["total_attempts"] == 3
    assert summary["total_retries"] == 2
    assert summary["retried_samples"] == 1
    run = summary["runs"][0]
    assert run["retry_count"] == 2
    assert "attempt_count" not in run
    assert "attempt_history" not in run
    assert Path(run["run_dir"]).parent == batch_dir
    assert [path for path in batch_dir.iterdir() if path.is_dir()] == [Path(run["run_dir"])]
    assert "transient fixture failure 3" in run["error"]


def test_prepare_resolves_both_projects_to_absolute_paths() -> None:
    prepared = prepare_pipeline(load_pipeline_config())
    assert prepared.config.run.sample_retries == 3
    simulator_model = Path(prepared.simulator_config.models["controller"])
    assistant_model = Path(prepared.assistant_config.models["assistant"])
    assert simulator_model.is_absolute()
    assert simulator_model.parent.name == "models"
    assert simulator_model.parent.parent.parent.name == "user_simulator"
    assert simulator_model.exists()
    assert assistant_model.is_absolute()
    assert assistant_model.parent.name == "models"
    assert assistant_model.parent.parent.parent.name == "assistant"
    assert assistant_model.exists()
    assert prepared.dataset.path.is_absolute()


def test_prepared_pipeline_batch_snapshot_records_resolved_models_and_prompts() -> None:
    prepared = prepare_pipeline(
        load_pipeline_config(),
        simulator_model_profile="deepseek_v4_flash_0731",
        assistant_model_profile="gpt_5_6_luna",
    )
    sample = one_node_sample()

    snapshot = prepared.batch_config_snapshot(
        batch_id="fixture-batch",
        created_at="2026-08-24T00:00:00+00:00",
        samples=[sample],
        concurrency=3,
    )

    assert next(iter(snapshot)) == "run_overview"
    assert snapshot["batch"]["sample_retries"] == 3
    assert snapshot["run_overview"] == {
        "difficulty": prepared.config.run.difficulty,
        "baseline": "base",
        "models": {
            "simulator": {
                key: {
                    "profile_name": "deepseek_v4_flash_0731",
                    "model_id": "deepseek/deepseek-v4-flash-0731",
                }
                for key in (
                    "controller",
                    "satisfaction",
                    "realizer_clear",
                    "realizer_abstract",
                )
            },
            "assistant": {
                "profile_name": "gpt_5_6_luna",
                "model_id": "openai/gpt-5.6-luna",
            },
        },
    }
    assert snapshot["simulator"]["model_profiles"]["controller"]["settings"]["model_id"] == (
        "deepseek/deepseek-v4-flash-0731"
    )
    assert snapshot["simulator"]["prompts"]["controller"]["system"]
    assert len(snapshot["simulator"]["prompts"]["controller"]["source_sha256"]) == 64
    assert snapshot["assistant"]["model_profile"]["settings"]["model_id"] == (
        "openai/gpt-5.6-luna"
    )
    assert snapshot["assistant"]["active_baseline"] == "base"
    assert snapshot["assistant"]["active_prompt"] is None


def test_local_vllm_assistant_does_not_require_assistant_openrouter_key() -> None:
    prepared = prepare_pipeline(
        load_pipeline_config(),
        assistant_model_profile="qwen_3_6_27b_vllm",
    )
    prepared.simulator_environment = SimulatorEnvironmentSettings(openrouter_api_key="test")
    prepared.assistant_environment = AssistantEnvironmentSettings(
        openrouter_api_key="",
        vllm_api_key="EMPTY",
    )

    assert prepared.missing_credentials == []
    assert prepared.has_api_key is True


def test_local_vllm_simulator_does_not_require_simulator_openrouter_key() -> None:
    prepared = prepare_pipeline(
        load_pipeline_config(),
        simulator_model_profile="qwen_3_8_27b_vllm",
    )
    prepared.simulator_environment = SimulatorEnvironmentSettings(
        openrouter_api_key="",
        vllm_api_key="EMPTY",
    )
    prepared.assistant_environment = AssistantEnvironmentSettings(openrouter_api_key="test")

    assert prepared.missing_credentials == []
    assert prepared.has_api_key is True
    profile = Path(prepared.simulator_config.models["controller"])
    assert profile.name == "qwen_3_8_27b_vllm.yaml"


def test_web_panel_streams_complete_message_events() -> None:
    html = _panel_html(
        sample_id="fixture",
        simulator_model="deepseek_v4_flash_0731.yaml",
        assistant_model="gpt_5_6_luna.yaml",
        baseline="base",
    )
    assert "EventSource" in html
    assert "event.event_type==='message'" in html
    assert "event.content" in html
    assert "delta" not in html
    assert '<link rel="icon" href="data:,">' in html
