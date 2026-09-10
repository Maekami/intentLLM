import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from assistant.config import GenerationSettings
from assistant.llm.base import GeneratedResponse

from interaction_pipeline import trace2skill
from interaction_pipeline.config import load_pipeline_config, resolve_pipeline_path
from interaction_pipeline.evolution_dataset import EvolutionDatasetInfo
from interaction_pipeline.prepare import prepare_pipeline
from interaction_pipeline.trace2skill import (
    FAILURE_TURN_LIMIT,
    INFRASTRUCTURE_FAILURE,
    SUCCESS,
    RunResult,
    VisibleMessage,
    VisibleTrajectory,
    canonical_visible_trajectory,
    classify_run_result,
    fixed_chunks,
    load_trace2skill_corpus,
    run_trace2skill_collection,
)
from interaction_pipeline.trace2skill_build import (
    ANALYSIS_ARTIFACT_TYPE,
    BUILD_PIPELINE,
    BUILD_SCHEMA_VERSION,
    OFFICIAL_TRACE2SKILL_COMMIT,
    AnalysisRecord,
    PatchEdit,
    SkillPatch,
    apply_patch_programmatically,
    balanced_chunks_upstream,
    build_analysis_messages,
    build_map_messages,
    call_patch_with_official_repair,
    merge_patch_group,
    order_combined_map_patches,
    parse_analysis_items,
    parse_patch_response,
    process_analysis_slot,
    process_map_slot,
    reduce_patches_hierarchically,
    run_trace2skill_build,
    sanitize_map_patches_for_single_file,
    sanitize_translated_edits,
    translate_final_patch,
    validate_runtime_skill,
    verify_and_fix_skill,
)
from interaction_pipeline.trace2skill_config import (
    Trace2SkillConfig,
    load_trace2skill_config,
    resolve_trace2skill_output_directory,
)


def _write_transcript(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def _patch(*, content: str = "- Clarify ambiguous requests before committing.") -> SkillPatch:
    return SkillPatch(
        reasoning="Consolidate a reusable dialogue lesson.",
        edits=[
            PatchEdit(
                file="SKILL.md",
                op="add_section",
                target_section="## Interaction Guidance",
                content=content,
            )
        ],
        changelog_entries=["Add interaction guidance"],
    )


def _patch_response(patch: SkillPatch | None = None) -> str:
    value = (patch or _patch()).model_dump(mode="json")
    return "```json\n" + json.dumps(value) + "\n```"


def test_canonical_failure_trajectory_drops_only_trailing_user(tmp_path) -> None:
    path = tmp_path / "transcript.jsonl"
    _write_transcript(
        path,
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply two"},
            {"role": "user", "content": "unmatched after budget"},
        ],
    )

    trajectory = canonical_visible_trajectory(
        path,
        outcome=FAILURE_TURN_LIMIT,
        expected_turns=2,
        turn_budget=2,
    )

    assert trajectory.assistant_turns == 2
    assert [item.content for item in trajectory.dialogue] == [
        "first",
        "reply one",
        "second",
        "reply two",
    ]


def test_run_classification_keeps_turn_limit_as_behavioral_failure() -> None:
    success = RunResult("a", "completed", 3, Path("success"))
    turn_limit = RunResult(
        "b",
        "failed",
        20,
        Path("failure"),
        "EpisodeTurnLimitError: reached budget",
    )
    infrastructure = RunResult(
        "c",
        "failed",
        7,
        Path("infra"),
        "ModelRequestError: timeout",
    )

    assert classify_run_result(success) == SUCCESS
    assert classify_run_result(turn_limit) == FAILURE_TURN_LIMIT
    assert classify_run_result(infrastructure) == INFRASTRUCTURE_FAILURE


def test_frozen_parallel_settings_match_audited_official_defaults() -> None:
    settings = load_trace2skill_config().run

    assert settings.map_batch_size == 1
    assert settings.merge_batch_size == 32
    assert settings.max_continuations == 2
    assert settings.format_fix_rounds == 2
    assert settings.max_merge_levels == 5
    assert settings.max_verification_rounds == 3
    assert settings.max_skill_lines == 500
    assert not hasattr(settings, "distillation_max_attempts")


def test_fixed_merge_topology_for_one_thousand_source_slots() -> None:
    chunks = fixed_chunks(list(range(1000)), 32)

    assert len(chunks) == 32
    assert [len(item) for item in chunks] == [32] * 31 + [8]
    assert [item for group in chunks for item in group] == list(range(1000))


def test_post_level_one_batches_use_official_small_tail_rebalancing() -> None:
    patches = [SkillPatch(reasoning=str(index)) for index in range(1000)]

    groups = balanced_chunks_upstream(patches, 32)

    assert len(groups) == 31
    assert sorted(map(len, groups)) == [32] * 23 + [33] * 8


def test_trace_config_rejects_non_frozen_development_shape() -> None:
    with pytest.raises(ValueError, match="exactly 1,000"):
        Trace2SkillConfig.model_validate(
            {
                "dataset": {
                    "expected_sample_count": 3,
                    "expected_difficulty_counts": {"easy": 1, "medium": 1, "hard": 1},
                }
            }
        )


def test_limit_output_directory_is_profile_and_trial_isolated(tmp_path) -> None:
    trial = resolve_trace2skill_output_directory(
        tmp_path,
        profile_name="model/profile",
        group_by_profile=True,
        limit=64,
    )
    canonical = resolve_trace2skill_output_directory(
        tmp_path,
        profile_name="qwen_3_6_27b_trace2skill",
        group_by_profile=True,
    )

    assert trial.name == "model_profile_trace2skill_smoke_n64"
    assert canonical.name == "qwen_3_6_27b_trace2skill"


def test_analysis_prompts_receive_visible_dialogue_only_and_parser_matches_official() -> None:
    trajectory = VisibleTrajectory(
        outcome=SUCCESS,
        dialogue=(
            VisibleMessage(role="user", content="private-visible-surface-token"),
            VisibleMessage(role="assistant", content="visible response"),
        ),
    )
    messages = build_analysis_messages(trajectory)
    serialized = json.dumps(messages)

    assert "private-visible-surface-token" in serialized
    assert "visible response" in serialized
    assert "sample_id" not in serialized
    assert "difficulty" not in serialized
    success_report = """# Lean Solution Path

## Overview
The assistant resolved the request.

## Step 1: Clarify
It clarified the visible ambiguity.

# Success Memory Item 1

## Title
Clarify Ambiguity

## Description
Resolve ambiguity before committing.

## Content
Ask a focused question when the visible request is underspecified.
"""
    failure_report = """# Failure Cause Item 1

## Title
Unresolved Ambiguity

## Description
The assistant committed too early.

## Content
The visible dialogue shows an unsupported assumption.

# Failure Memory Item 1

## Title
Clarify First

## Description
Resolve ambiguity before committing.

## Content
Ask a focused question before selecting one interpretation.
"""

    success_items = parse_analysis_items(success_report, outcome=SUCCESS)
    failure_items = parse_analysis_items(failure_report, outcome=FAILURE_TURN_LIMIT)
    assert [item.type for item in success_items] == ["success_memory"]
    assert [item.type for item in failure_items] == ["failure_cause", "failure_memory"]
    map_messages = build_map_messages(AnalysisRecord(record_source="success", items=success_items))
    assert "private-visible-surface-token" not in json.dumps(map_messages)
    assert "Clarify Ambiguity" in json.dumps(map_messages)
    assert "Adjust Degrees of Freedom from Both Signals" in map_messages[0]["content"]
    assert "Prefer Evidence-Weighted Concision" in map_messages[0]["content"]


def test_patch_parser_ports_official_malformed_json_recovery() -> None:
    malformed = """```json
{"reasoning":"Keep the lesson","edits":[{"file":"SKILL.md","op":"add_section","target_section":"## Guidance","content":"- Clarify first."}],"changelog_entries":["Added guidance",]}
```"""

    patches, feedback = parse_patch_response(malformed)

    assert feedback == ""
    assert len(patches) == 1
    assert patches[0].edits[0].file == "SKILL.md"


def test_combined_map_order_matches_official_and_compacts_exclusions() -> None:
    outcomes = [SUCCESS, FAILURE_TURN_LIMIT, SUCCESS, FAILURE_TURN_LIMIT]
    trajectories = [
        VisibleTrajectory(
            outcome=outcome,
            dialogue=(
                VisibleMessage(role="user", content=f"request {index}"),
                VisibleMessage(role="assistant", content=f"response {index}"),
            ),
        )
        for index, outcome in enumerate(outcomes)
    ]
    slots = [
        {
            "status": "complete",
            "sample_id": sample_id,
            "patch": SkillPatch(reasoning=str(index)).model_dump(mode="json"),
        }
        for index, sample_id in enumerate(("z", "z", "a", "a"))
    ]
    slots[1] = {"status": "skipped", "sample_id": "z", "patch": None}

    patches, indices = order_combined_map_patches(slots, trajectories)

    assert indices == [3, 2, 0]
    assert [patch.reasoning for patch in patches] == ["3", "2", "0"]


class SequenceClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.last_call_metadata: dict = {}

    async def generate(self, *, messages, generation):
        self.calls.append([dict(item) for item in messages])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return GeneratedResponse(content=response, metadata={"model_id": "fixture"})


async def test_official_repair_uses_continuation_for_unclosed_json() -> None:
    client = SequenceClient(
        [
            '```json\n{"reasoning":"continued",',
            '"edits":[],"changelog_entries":[]}\n```',
        ]
    )

    patches, calls, conversation, _ = await call_patch_with_official_repair(
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
    )

    assert len(patches) == 1
    assert [item["kind"] for item in calls] == ["initial", "continuation"]
    assert conversation[-1]["role"] == "assistant"


async def test_official_repair_skips_continuation_for_closed_but_invalid_json() -> None:
    client = SequenceClient(
        [
            '```json\n{"reasoning":7}\n```',
            _patch_response(SkillPatch(reasoning="fixed", edits=[], changelog_entries=[])),
        ]
    )

    patches, calls, _, _ = await call_patch_with_official_repair(
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
    )

    assert len(patches) == 1
    assert [item["kind"] for item in calls] == ["initial", "format_fix"]
    assert "Missing top-level field: edits" in client.calls[1][-1]["content"]


async def test_analysis_zero_items_is_excluded_instead_of_blocking(tmp_path) -> None:
    client = SequenceClient(["# No Causally Supported Failure\nThe dialogue is inconclusive."])
    trajectory = VisibleTrajectory(
        outcome=FAILURE_TURN_LIMIT,
        dialogue=(
            VisibleMessage(role="user", content="hello"),
            VisibleMessage(role="assistant", content="hello"),
        ),
    )

    artifact = await process_analysis_slot(
        artifact_path=tmp_path / "analysis.json",
        dataset_index=0,
        sample_id="private-id",
        source_trajectory_sha256="source-hash",
        trajectory=trajectory,
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
    )

    assert artifact["status"] == "skipped"
    assert artifact["skip_reason"] == "no_analysis_items_parsed"
    assert artifact["record"] is None
    assert [message["role"] for message in artifact["conversation"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert "hello" in artifact["conversation"][1]["content"]


async def test_map_request_failure_is_excluded_instead_of_pending(tmp_path) -> None:
    source = {
        "status": "complete",
        "record": AnalysisRecord(
            record_source="success",
            items=[
                {
                    "type": "success_memory",
                    "number": 1,
                    "title": "Clarify",
                    "description": "Clarify ambiguity.",
                    "content": "Ask a focused question.",
                }
            ],
        ).model_dump(mode="json"),
    }
    client = SequenceClient([RuntimeError("provider unavailable")])

    artifact = await process_map_slot(
        artifact_path=tmp_path / "map.json",
        dataset_index=0,
        sample_id="private-id",
        source_analysis=source,
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        settings=load_trace2skill_config().run,
    )

    assert artifact["status"] == "skipped"
    assert artifact["patch"] is None
    assert artifact["skip_reason"] == "map_patch_unavailable_after_official_repair"


async def test_merge_failure_retains_original_group(tmp_path) -> None:
    patches = [_patch(content="- First."), _patch(content="- Second.")]
    client = SequenceClient([RuntimeError("merge unavailable")])
    artifact_path = tmp_path / "merge.json"

    output = await merge_patch_group(
        artifact_path=artifact_path,
        level=1,
        group_index=0,
        patches=patches,
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        settings=load_trace2skill_config().run,
    )

    assert output == patches
    artifact = json.loads(artifact_path.read_text())
    assert artifact["status"] == "fallback"
    assert artifact["fallback"] == "retain_input_patches"


class FailingPatchClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_call_metadata: dict = {}

    async def generate(self, *, messages, generation):
        self.calls += 1
        raise RuntimeError("merge unavailable")


async def test_reduce_uses_official_forced_merge_then_first_patch_fallback(tmp_path) -> None:
    source = [_patch(content="- First."), _patch(content="- Second.")]
    client = FailingPatchClient()

    result, summary = await reduce_patches_hierarchically(
        run_dir=tmp_path,
        source_ordered_patches=source,
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        settings=load_trace2skill_config().run,
    )

    assert result == source[0]
    assert client.calls == 6
    assert summary["forced_merge"] is True
    assert summary["ultimate_first_patch_fallback"] is True
    assert len(summary["levels"]) == 6


class PatchClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.last_call_metadata: dict = {}

    async def generate(self, *, messages, generation):
        self.calls.append([dict(item) for item in messages])
        return GeneratedResponse(content=_patch_response(), metadata={"model_id": "fixture"})


async def test_reduce_uses_thirty_two_source_groups_then_one_final_merge(tmp_path) -> None:
    client = PatchClient()
    source = [_patch() for _ in range(1000)]

    result, summary = await reduce_patches_hierarchically(
        run_dir=tmp_path,
        source_ordered_patches=source,
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        settings=load_trace2skill_config().run,
    )

    assert result is not None
    assert len(client.calls) == 33
    first = sorted((tmp_path / "merge_levels" / "level_01").glob("group_*.json"))
    second = sorted((tmp_path / "merge_levels" / "level_02").glob("group_*.json"))
    assert len(first) == 32
    assert len(second) == 1
    assert json.loads(first[-1].read_text())["input_patch_count"] == 8
    assert summary["levels"][0]["output_patch_count"] == 32
    assert summary["levels"][1]["input_patch_count"] == 32


def test_programmatic_apply_ports_official_markdown_operations() -> None:
    patch = SkillPatch(
        reasoning="Apply exact edits.",
        edits=[
            PatchEdit(
                file="SKILL.md",
                op="add_section",
                target_section="## Guidance",
                content="- First.",
            ),
            PatchEdit(
                file="SKILL.md",
                op="append_to_section",
                target_section="## Guidance",
                content="- Second.",
            ),
            PatchEdit(
                file="SKILL.md",
                op="replace_in_section",
                target_section="## Guidance",
                old_text="First",
                content="Updated",
            ),
        ],
        changelog_entries=[],
    )

    content, records = apply_patch_programmatically("", patch)

    assert "## Guidance" in content
    assert "- Updated." in content
    assert "- Second." in content
    assert [item["status"] for item in records] == ["applied", "applied", "applied"]


def test_sanitizer_only_adapts_official_paths_to_single_runtime_file() -> None:
    valid = PatchEdit(file="SKILL.md", op="add_section", content="x")
    reference = PatchEdit(file="references/detail.md", op="add_section", content="x")
    unsupported = PatchEdit(file="SKILL.md", op="create", content="x")
    dangling_link = PatchEdit(
        file="SKILL.md",
        op="add_section",
        content="Read references/detail.md.",
    )

    kept, dropped = sanitize_translated_edits([valid, reference, unsupported, dangling_link])

    assert kept == [valid]
    assert [item["reason"] for item in dropped] == [
        "outside_single_file_skill",
        "unsupported_operation",
        "unsupported_reference_link",
    ]


def test_map_create_link_pairing_is_adapted_to_single_file_layout() -> None:
    patch = SkillPatch(
        edits=[
            PatchEdit(
                file="SKILL.md",
                op="add_section",
                content="Read references/detail.md.",
            ),
            PatchEdit(file="references/detail.md", op="create_file", content="# Detail"),
            PatchEdit(file="SKILL.md", op="add_section", content="- Keep this guidance."),
        ]
    )

    patches, records = sanitize_map_patches_for_single_file([patch], [7])

    assert [edit.content for edit in patches[0].edits] == ["- Keep this guidance."]
    assert patch.edits[1].op == "create"
    assert records[0]["dataset_index"] == 7
    assert [item["reason"] for item in records[0]["dropped_edits"]] == [
        "unsupported_reference_link",
        "outside_single_file_skill",
    ]


async def test_translation_failure_falls_back_to_original_edit(tmp_path) -> None:
    client = SequenceClient([RuntimeError("translation unavailable")])
    source = _patch()

    translated, summary = await translate_final_patch(
        run_dir=tmp_path,
        patch=source,
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        settings=load_trace2skill_config().run,
    )

    assert translated.edits == source.edits
    assert summary["fallback_count"] == 1
    assert (
        json.loads((tmp_path / "translation" / "edit_0001.json").read_text())["fallback"]
        == "use_original_edit"
    )


async def test_translation_passes_file_level_edits_without_an_llm_call(tmp_path) -> None:
    client = SequenceClient([])
    source = SkillPatch(
        edits=[PatchEdit(file="references/detail.md", op="create", content="# Detail")]
    )

    translated, summary = await translate_final_patch(
        run_dir=tmp_path,
        patch=source,
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        settings=load_trace2skill_config().run,
    )

    assert client.calls == []
    assert translated.edits == []
    assert summary["passthrough_count"] == 1
    assert summary["dropped_edits"][0]["reason"] == "unsupported_operation"


async def test_validator_runs_official_style_patch_fix_loop(tmp_path) -> None:
    client = PatchClient()

    content, rounds, valid, message = await verify_and_fix_skill(
        run_dir=tmp_path,
        skill_content="",
        client=client,
        generation=GenerationSettings(max_completion_tokens=100),
        settings=load_trace2skill_config().run,
    )

    assert valid is True
    assert message == "Skill is valid"
    assert len(rounds) == 1
    assert "## Interaction Guidance" in content
    assert validate_runtime_skill(content, max_lines=500)[0] is True


class RecordingPrepared:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def build(self, sample, *, output_dir, seed, run_id, **kwargs):
        self.seeds.append(seed)
        run_dir = Path(output_dir) / run_id
        run_dir.mkdir()
        (run_dir / "transcript.jsonl").touch()
        return SimpleNamespace(
            episode=SimpleNamespace(),
            assistant=SimpleNamespace(),
            audit=SimpleNamespace(run_dir=run_dir),
        )


async def test_collection_slot_retries_only_infrastructure_with_same_seed(
    monkeypatch, tmp_path
) -> None:
    prepared = RecordingPrepared()
    calls = 0

    async def fake_interaction(*, audit, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return RunResult(
                "private-id",
                "failed",
                0,
                audit.run_dir,
                "ModelRequestError: timeout",
            )
        _write_transcript(
            audit.run_dir / "transcript.jsonl",
            [
                {"role": "user", "content": "visible request"},
                {"role": "assistant", "content": "visible response"},
            ],
        )
        return RunResult("private-id", "completed", 1, audit.run_dir)

    monkeypatch.setattr(trace2skill, "run_interaction", fake_interaction)
    slot = await trace2skill._collect_trajectory_slot(
        prepared=prepared,
        sample=SimpleNamespace(sample_id="private-id"),
        dataset_index=4,
        difficulty="hard",
        run_dir=tmp_path,
        slot_path=tmp_path / "slot.json",
        settings=SimpleNamespace(
            seed=101,
            infrastructure_max_attempts=3,
            turn_budget=20,
        ),
        dataset_sha256="dataset-hash",
    )

    assert prepared.seeds == [105, 105]
    assert slot["status"] == "complete"
    assert [item["classification"] for item in slot["episode_attempts"]] == [
        INFRASTRUCTURE_FAILURE,
        SUCCESS,
    ]
    assert "analysis" not in slot


def _write_bound_trace_profile(tmp_path: Path, *, skill_path: Path) -> Path:
    source_path = resolve_pipeline_path(
        "../assistant/configs/models/qwen_3_6_27b_vllm_non_thinking.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["profile_name"] = "fixture_qwen_trace2skill"
    raw["memory"] = None
    raw["skill"] = {"framework": "trace2skill", "path": str(skill_path)}
    profile_path = tmp_path / "fixture_qwen_trace2skill.yaml"
    profile_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return profile_path


def _mock_full_development_inputs(*, assistant_model_profile: str):
    config = load_trace2skill_config(
        overrides={"run": {"trajectory_concurrency": 32, "merge_concurrency": 8}}
    )
    dataset_path = resolve_pipeline_path(config.dataset.path)
    original_path = resolve_pipeline_path(config.dataset.original_path)
    metadata_path = resolve_pipeline_path(config.dataset.metadata_path)
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "path": str(dataset_path),
                    "original_path": str(original_path),
                    "metadata_path": str(metadata_path),
                }
            )
        }
    )
    prepared = prepare_pipeline(
        load_pipeline_config(),
        assistant_baseline="base",
        assistant_model_profile=assistant_model_profile,
        assistant_profile_skill_enabled=False,
        max_turns=20,
        simulator_dataset_path=dataset_path,
    )
    sample_ids = tuple(f"fixture-{index:04d}" for index in range(1000))
    samples = [
        SimpleNamespace(
            sample_id=sample_id,
            model_extra={
                "difficulty": "easy" if index < 334 else "medium" if index < 667 else "hard"
            },
        )
        for index, sample_id in enumerate(sample_ids)
    ]
    dataset_info = EvolutionDatasetInfo(
        path=dataset_path,
        sha256="dev-sha256",
        original_path=original_path,
        original_sha256="original-sha256",
        metadata_path=metadata_path,
        metadata_sha256="metadata-sha256",
        sample_ids=sample_ids,
        difficulty_field="difficulty",
        difficulty_counts={"easy": 334, "medium": 333, "hard": 333},
        assignment_seed=42,
        assignment_algorithm="sha256_seeded_ranking_v1",
        ordered_assignments_sha256="assignment-sha256",
    )
    return config, prepared, samples, dataset_info


async def _fake_collected_slot(
    *,
    dataset_index,
    sample,
    dataset_sha256,
    slot_path,
    **kwargs,
):
    trajectory = VisibleTrajectory(
        outcome=SUCCESS,
        dialogue=(
            VisibleMessage(role="user", content=f"visible request token {dataset_index}"),
            VisibleMessage(role="assistant", content="visible response"),
        ),
    )
    trajectory_sha256 = trace2skill._trajectory_entry_hash(
        dataset_index=dataset_index,
        sample_id=sample.sample_id,
        dataset_sha256=dataset_sha256,
        trajectory=trajectory,
    )
    slot = {
        "schema_version": 1,
        "artifact_type": trace2skill.COLLECTION_SLOT_ARTIFACT_TYPE,
        "dataset_index": dataset_index,
        "sample_id": sample.sample_id,
        "dataset_sha256": dataset_sha256,
        "status": "complete",
        "outcome": SUCCESS,
        "turns": 1,
        "visible_trajectory": trajectory.model_dump(mode="json"),
        "trajectory_sha256": trajectory_sha256,
        "episode_attempts": [],
    }
    Path(slot_path).write_text(json.dumps(slot), encoding="utf-8")
    return slot


async def _create_fake_collection(monkeypatch, tmp_path, *, profile_path):
    config, prepared, samples, dataset_info = _mock_full_development_inputs(
        assistant_model_profile=str(profile_path)
    )
    monkeypatch.setattr(trace2skill, "_collect_trajectory_slot", _fake_collected_slot)
    result = await run_trace2skill_collection(
        prepared,
        samples,
        output_root=tmp_path / "collections",
        trace_config=config,
        dataset_info=dataset_info,
    )
    return result, config


class OfficialPipelineClient:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.last_call_metadata: dict = {}

    async def generate(self, *, messages, generation):
        self.calls.append([dict(item) for item in messages])
        system = messages[0]["content"]
        if "successful AI-agent trajectory analysis" in system:
            content = """# Lean Solution Path

## Overview
The assistant handled the visible request.

## Step 1: Respond
It responded to the user.

# Success Memory Item 1

## Title
Clarify Ambiguity

## Description
Clarify an underspecified request.

## Content
Ask a focused question before committing to one interpretation.
"""
        elif "failure-analysis agent" in system:
            content = """# Failure Cause Item 1

## Title
Unresolved Ambiguity

## Description
The assistant committed prematurely.

## Content
The visible dialogue showed an unsupported assumption.

# Failure Memory Item 1

## Title
Clarify First

## Description
Resolve visible ambiguity.

## Content
Ask a focused question before choosing one interpretation.
"""
        else:
            content = _patch_response()
        return GeneratedResponse(content=content, metadata={"model_id": "fixture"})


async def test_decoupled_trial_build_is_complete_auditable_and_resumable(
    monkeypatch, tmp_path
) -> None:
    profile_path = _write_bound_trace_profile(
        tmp_path,
        skill_path=tmp_path / "must_not_publish" / "SKILL.md",
    )
    collection, config = await _create_fake_collection(
        monkeypatch,
        tmp_path,
        profile_path=profile_path,
    )
    corpus = load_trace2skill_corpus(collection.run_dir)
    assert len(corpus.trajectories) == 1000
    collection_bytes = {
        path.relative_to(collection.run_dir): path.read_bytes()
        for path in collection.run_dir.rglob("*")
        if path.is_file()
    }
    client = OfficialPipelineClient()

    result = await run_trace2skill_build(
        collection_dir=collection.run_dir,
        output_root=tmp_path / "builds",
        trace_config=config,
        profile_path=profile_path,
        analyst_client=client,
        limit=64,
    )

    assert result.complete is True
    assert result.canonical is False
    assert result.patch_count == 64
    assert result.skipped_count == 0
    assert result.final_skill_path == result.run_dir / "SMOKE_SKILL.md"
    assert result.final_skill_path.exists()
    assert not (tmp_path / "must_not_publish" / "SKILL.md").exists()
    assert len(client.calls) == 132  # 64 analysis + 64 MAP + 3 merge + 1 translation
    assert len(list((result.run_dir / "trajectory_analysis").glob("*.json"))) == 64
    assert len(list((result.run_dir / "map_patches").glob("*.json"))) == 64
    assert len(list((result.run_dir / "merge_levels" / "level_01").glob("*.json"))) == 2
    assert len(list((result.run_dir / "merge_levels" / "level_02").glob("*.json"))) == 1
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == BUILD_SCHEMA_VERSION
    assert manifest["pipeline"] == BUILD_PIPELINE
    assert manifest["official_implementation"]["commit"] == OFFICIAL_TRACE2SKILL_COMMIT
    assert manifest["necessary_task_adaptations"]["trajectory_evidence"].startswith("visible")
    assert manifest["official_alignment"]["apply"].startswith("deterministic")
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["analysis_skip_reasons"] == {}
    assert summary["map_skip_reasons"] == {}
    assert summary["interaction_call_count"] == 0
    assert summary["logical_llm_call_count"] == 132
    assert all("fixture-" not in json.dumps(messages) for messages in client.calls)
    assert {
        path.relative_to(collection.run_dir): path.read_bytes()
        for path in collection.run_dir.rglob("*")
        if path.is_file()
    } == collection_bytes

    result.final_skill_path.unlink()
    resumed = await run_trace2skill_build(
        collection_dir=collection.run_dir,
        output_root=tmp_path / "builds",
        trace_config=config,
        profile_path=profile_path,
        analyst_client=client,
        resume_dir=result.run_dir,
        limit=64,
    )
    assert resumed.complete is True
    assert result.final_skill_path.exists()
    assert len(client.calls) == 132


async def test_old_custom_build_cannot_be_resumed_into_aligned_pipeline(
    monkeypatch, tmp_path
) -> None:
    profile_path = _write_bound_trace_profile(
        tmp_path,
        skill_path=tmp_path / "published" / "SKILL.md",
    )
    collection, config = await _create_fake_collection(
        monkeypatch,
        tmp_path,
        profile_path=profile_path,
    )
    old = tmp_path / "old_build"
    old.mkdir()
    (old / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": trace2skill.BUILD_ARTIFACT_TYPE,
                "immutable_fingerprint": "old",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="predates the official-aligned"):
        await run_trace2skill_build(
            collection_dir=collection.run_dir,
            output_root=tmp_path / "builds",
            trace_config=config,
            profile_path=profile_path,
            analyst_client=OfficialPipelineClient(),
            resume_dir=old,
            limit=64,
        )


async def test_canonical_build_publishes_profile_specific_skill_with_corpus_provenance(
    monkeypatch, tmp_path
) -> None:
    published = tmp_path / "published" / "SKILL.md"
    profile_path = _write_bound_trace_profile(tmp_path, skill_path=published)
    collection, config = await _create_fake_collection(
        monkeypatch,
        tmp_path,
        profile_path=profile_path,
    )

    async def fake_analysis(
        *, artifact_path, dataset_index, sample_id, source_trajectory_sha256, **kwargs
    ):
        record = AnalysisRecord(
            record_source="success",
            items=[
                {
                    "type": "success_memory",
                    "number": 1,
                    "title": "Clarify",
                    "description": "Clarify ambiguity.",
                    "content": "Ask a focused question.",
                }
            ],
        )
        artifact = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "artifact_type": ANALYSIS_ARTIFACT_TYPE,
            "dataset_index": dataset_index,
            "sample_id": sample_id,
            "source_trajectory_sha256": source_trajectory_sha256,
            "status": "complete",
            "calls": [],
            "record": record.model_dump(mode="json"),
        }
        Path(artifact_path).write_text(json.dumps(artifact), encoding="utf-8")
        return artifact

    async def fake_map(*, artifact_path, dataset_index, sample_id, source_analysis, **kwargs):
        artifact = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "dataset_index": dataset_index,
            "sample_id": sample_id,
            "status": "complete",
            "calls": [],
            "patch": _patch().model_dump(mode="json"),
        }
        Path(artifact_path).write_text(json.dumps(artifact), encoding="utf-8")
        return artifact

    monkeypatch.setattr(
        "interaction_pipeline.trace2skill_build.process_analysis_slot", fake_analysis
    )
    monkeypatch.setattr("interaction_pipeline.trace2skill_build.process_map_slot", fake_map)
    client = PatchClient()
    result = await run_trace2skill_build(
        collection_dir=collection.run_dir,
        output_root=tmp_path / "builds",
        trace_config=config,
        profile_path=profile_path,
        analyst_client=client,
    )

    assert result.complete is True
    assert result.canonical is True
    assert result.published_skill_path == published
    assert published.read_text() == result.final_skill_path.read_text()
    assert len(client.calls) == 34  # 32 first-level + 1 final merge + 1 translation
    provenance = json.loads((published.parent / "SKILL.provenance.json").read_text())
    assert provenance["source_trajectory_collection_dir"] == str(collection.run_dir)
    assert provenance["source_trajectory_corpus_sha256"] == collection.corpus_sha256
    assert provenance["development_sample_count"] == 1000

    resumed = await run_trace2skill_build(
        collection_dir=collection.run_dir,
        output_root=tmp_path / "builds",
        trace_config=config,
        profile_path=profile_path,
        analyst_client=client,
        resume_dir=result.run_dir,
    )
    assert resumed.complete is True
    assert len(client.calls) == 34
    assert json.loads((published.parent / "SKILL.provenance.json").read_text()) == provenance
