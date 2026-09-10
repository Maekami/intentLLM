import inspect
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from user_simulator.audit.logger import AuditLogger
from user_simulator.cli import demo, prompt_smoke_test
from user_simulator.config import EnvironmentSettings, load_config, load_model_profile
from user_simulator.controller.llm_controller import LLMController
from user_simulator.domain.enums import Difficulty, RealizationMode
from user_simulator.domain.results import (
    ControllerResult,
    SatisfactionUpdateResult,
    UserGenerationResult,
)
from user_simulator.engine.episode import Episode
from user_simulator.exceptions import ConfigurationError
from user_simulator.factory import (
    CONTROLLER_REGISTRY,
    REALIZER_REGISTRY,
    SATISFACTION_REGISTRY,
    ComponentSpec,
    build_simulator_components,
    configured_components_require_openrouter,
)
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.llm.schema_registry import SCHEMA_REGISTRY, resolve_schema
from user_simulator.mock_components import (
    MockController,
    MockSatisfactionUpdater,
    MockUserRealizer,
)

PROMPT_PATHS = [
    "configs/prompts/controller.yaml",
    "configs/prompts/satisfaction.yaml",
    "configs/prompts/user_clear.yaml",
    "configs/prompts/user_abstract.yaml",
]


@pytest.mark.parametrize("path", PROMPT_PATHS)
def test_canonical_prompt_contract(path: str) -> None:
    prompt = PromptTemplate.load(path)
    assert prompt.schema is resolve_schema(prompt.schema_name)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert set(raw) == {"name", "schema_name", "system", "user_template"}


def test_prompt_hash_changes_with_rendered_content() -> None:
    prompt = PromptTemplate.load("configs/prompts/controller.yaml")
    _, first = prompt.render(context="first", candidate_order='["N2"]', correction="")
    _, second = prompt.render(context="second", candidate_order='["N2"]', correction="")
    assert first["prompt_hash"] != second["prompt_hash"]
    assert first["schema_hash"] == prompt.schema.hash
    assert first["prompt_path"] == prompt.path
    assert set(first) == {
        "prompt_name",
        "prompt_path",
        "prompt_hash",
        "schema_name",
        "schema_hash",
    }


def test_critical_controller_and_satisfaction_semantics_are_explicit() -> None:
    controller = PromptTemplate.load("configs/prompts/controller.yaml").system
    satisfaction = PromptTemplate.load("configs/prompts/satisfaction.yaml").system
    assert "Semantic relatedness alone is not enough" in controller
    assert "targeted clarification question" in controller
    assert "hidden-node-blind evidence rule" in controller
    assert "candidate metadata only to define the semantic target" in controller
    assert "even if the user would not need to restate" in controller
    assert "DIRECT-NEXT-REPLY REQUIREMENT" in controller
    assert "For Tests 2 and 3" in controller
    assert "Never use a hypothetical intermediate user turn" in controller
    assert "specific hobby such as movies" not in controller
    assert "directly trigger" in controller
    assert "PREFIX CLOSURE" in controller
    assert "You decide intent exposure only" in controller
    assert "Graph-terminal exposure" in controller
    assert "end_exposed" not in controller
    assert "END EXPOSURE" not in controller
    assert "user's own" in satisfaction
    assert "is not evidence that the intent has been satisfied" in " ".join(satisfaction.split())
    assert all(
        label in satisfaction for label in ("unsatisfied", "partially_satisfied", "satisfied")
    )
    assert "Never evaluate END" in satisfaction
    assert "STOP-NOW TEST" in satisfaction
    assert "Progress toward obtaining a future answer" in satisfaction
    assert "interaction-style node" in satisfaction
    assert "does not need to have explicitly verbalized" in satisfaction
    assert "Do not penalize the assistant merely because it proactively inferred" in satisfaction
    assert "usable but overly general assistance" in satisfaction
    assert "promising but overly general" not in satisfaction
    assert "REQUIREMENT-SOURCE BOUNDARY" in satisfaction
    assert "An assistant statement that information X is necessary" in satisfaction
    assert "cannot create a new required deliverable" in satisfaction
    assert "REMAINING-GAP VALIDITY TEST" in satisfaction
    assert "optional personalization" in satisfaction
    assert "later or unexposed node" in satisfaction
    assert "not an independent source of task facts" in satisfaction
    assert "If the unresolved need is broad, keep the gap broad" in satisfaction
    assert "remaining_gap" in satisfaction
    assert "must be null" in satisfaction


@pytest.mark.parametrize(
    "path",
    ["configs/prompts/user_clear.yaml", "configs/prompts/user_abstract.yaml"],
)
def test_human_realizer_prompt_requirements(path: str) -> None:
    system = PromptTemplate.load(path).system
    normalized = " ".join(system.lower().split())
    required = [
        "human user",
        "speak only as the user",
        "shortest natural message" if "clear" in path else "does not volunteer",
        "apparent knowledge level",
        "write every user message in english only",
        "visible conversation",
        "do not invent credentials",
        "preferences",
        "keep the message focused on the task",
        "small minority of turns",
        "do not force an imperfection",
        "unsupported task-relevant content",
        "unselected unresolved nodes",
        "initial turn",
        "partially_satisfied",
        "task summary and task expectation",
        "global direction and consistency only",
        "keep every unspecified slot unspecified",
        "expresses all and only the selected unresolved intents",
        "already-established user facts from the visible conversation",
        "any acceptance, rejection, correction, or preference",
        "otherwise, refer to the proposal neutrally",
        "not an independent factual source",
        "ignore any unsupported detail in a gap",
        "factual permission boundary always takes precedence",
        "model-reported coverage check",
        "model-reported unsupported-task-content check",
    ]
    for phrase in required:
        assert phrase in normalized
    assert "selected node remaining gaps" in normalized
    assert (
        "abstraction may reduce specificity, but must never add specificity" in normalized
        if "abstract" in path
        else "clarity may unpack selected-node content, but must never add new specificity"
        in normalized
    )
    if "abstract" in path:
        assert "derive each diagnostic clue only from information authorized" in normalized
        assert "broad but recognizable clue" in normalized


def test_satisfaction_schema_encodes_status_remaining_gap_combinations() -> None:
    schema = resolve_schema("satisfaction_update_result").json_schema
    item_schema = schema["properties"]["updates"]["items"]
    variant_names = {item["$ref"].split("/")[-1] for item in item_schema["anyOf"]}
    assert variant_names == {
        "UnsatisfiedNodeSatisfactionDecision",
        "PartiallySatisfiedNodeSatisfactionDecision",
        "SatisfiedNodeSatisfactionDecision",
    }
    expected = {
        "UnsatisfiedNodeSatisfactionDecision": ("unsatisfied", "null"),
        "PartiallySatisfiedNodeSatisfactionDecision": ("partially_satisfied", "string"),
        "SatisfiedNodeSatisfactionDecision": ("satisfied", "null"),
    }
    for name, (status, gap_type) in expected.items():
        variant = schema["$defs"][name]
        assert {"status", "remaining_gap"} <= set(variant["required"])
        assert variant["properties"]["status"]["const"] == status
        assert variant["properties"]["remaining_gap"]["type"] == gap_type
    partial_gap = schema["$defs"]["PartiallySatisfiedNodeSatisfactionDecision"][
        "properties"
    ]["remaining_gap"]
    assert partial_gap["minLength"] == 1
    assert "explicitly required by this node" in partial_gap["description"]
    assert "assistant-introduced prerequisites" in partial_gap["description"]
    assert "later-node information" in partial_gap["description"]


def test_failed_structured_validation_marks_semantic_validation_not_run() -> None:
    episode = object.__new__(Episode)
    episode.audit = SimpleNamespace(git_commit="test")
    component = SimpleNamespace(
        last_call_metadata={"structured_validation_status": "invalid"},
        semantic_events=[],
    )

    metadata = episode._llm_metadata(component, "satisfaction")

    assert metadata["semantic_validation_status"] == "not_run"
    assert metadata["semantic_retry_count"] == 0


def test_llm_token_breakdown_is_preserved_in_episode_audit_metadata() -> None:
    episode = object.__new__(Episode)
    episode.audit = SimpleNamespace(git_commit="test")
    component = SimpleNamespace(
        last_call_metadata={
            "input_tokens": 11,
            "output_tokens": 17,
            "thinking_tokens": 7,
            "answer_tokens": 10,
            "structured_validation_status": "valid",
        },
        semantic_events=[],
    )

    metadata = episode._llm_metadata(component, "controller")

    assert metadata["input_tokens"] == 11
    assert metadata["output_tokens"] == 17
    assert metadata["thinking_tokens"] == 7
    assert metadata["answer_tokens"] == 10


def test_unknown_prompt_field_and_placeholder_fail_at_load(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
name: bad
schema_name: controller_result
manual_revision: 2
system: test
user_template: "{context} {candidate_order} {correction} {unknown}"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown prompt fields"):
        PromptTemplate.load(path)


def test_schema_registry_is_canonical_and_closed() -> None:
    assert set(SCHEMA_REGISTRY) == {
        "controller_result",
        "satisfaction_update_result",
        "user_generation_result",
    }
    assert resolve_schema("controller_result").model is ControllerResult
    assert resolve_schema("satisfaction_update_result").model is SatisfactionUpdateResult
    assert resolve_schema("user_generation_result").model is UserGenerationResult
    for spec in SCHEMA_REGISTRY.values():
        schema = spec.json_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]
        assert len(spec.hash) == 64


def test_schema_hash_is_stable_for_unchanged_model() -> None:
    first = resolve_schema("controller_result").hash
    second = resolve_schema("controller_result").hash
    assert first == second


def test_user_schema_has_explicit_model_reported_checks_and_enum() -> None:
    schema = resolve_schema("user_generation_result").json_schema
    assert "coverage" in schema["properties"]
    assert schema["properties"]["selected_node_ids"]["minItems"] == 1
    assert schema["properties"]["coverage"]["minItems"] == 1
    assert "contains_unsupported_task_content" in schema["properties"]
    assert "contains_unsupported_intent" not in schema["properties"]
    mode_ref = schema["properties"]["realization_mode"]["$ref"].split("/")[-1]
    assert schema["$defs"][mode_ref]["enum"] == ["clear", "abstract"]


def test_user_schema_rejects_empty_coverage_for_selected_node() -> None:
    with pytest.raises(ValidationError, match="coverage"):
        UserGenerationResult.model_validate(
            {
                "user_message": "What should I do next?",
                "selected_node_ids": ["N1"],
                "realization_mode": "clear",
                "coverage": [],
                "contains_unsupported_task_content": False,
                "summary": "invalid empty coverage",
            }
        )


def test_factory_wires_mode_specific_realizer_semantic_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_profile = yaml.safe_load(
        Path("configs/models/deepseek_v4_flash_0731.yaml").read_text(encoding="utf-8")
    )
    profile_paths: dict[str, Path] = {}
    for mode, attempts in (("clear", 2), ("abstract", 4)):
        profile = {**base_profile, "profile_name": f"test_{mode}"}
        profile["retry"] = {**base_profile["retry"], "max_attempts": attempts}
        path = tmp_path / f"{mode}.yaml"
        path.write_text(yaml.safe_dump(profile), encoding="utf-8")
        profile_paths[mode] = path

    config = load_config(
        cli_overrides={
            "models": {
                "realizer_clear": str(profile_paths["clear"]),
                "realizer_abstract": str(profile_paths["abstract"]),
            }
        }
    )
    monkeypatch.setattr(
        "user_simulator.factory.OpenRouterStructuredClient",
        lambda *args, **kwargs: object(),
    )
    components = build_simulator_components(
        config=config,
        environment=EnvironmentSettings(openrouter_api_key="test"),
    )

    assert components.user_realizer.semantic_attempts == {
        RealizationMode.CLEAR: 2,
        RealizationMode.ABSTRACT: 4,
    }


def test_structured_settings_cannot_disable_strictness() -> None:
    with pytest.raises(ValidationError):
        load_model_profile(
            overrides={
                "structured_output": {
                    "type": "json_schema",
                    "strict": False,
                    "require_parameters": True,
                }
            }
        )


def _mock_config(policy: dict[str, int] | None = None):
    overrides = {
        "components": {
            "controller": "mock_controller",
            "satisfaction_updater": "mock_satisfaction_updater",
            "user_realizer": "mock_user_realizer",
        }
    }
    if policy:
        overrides["policy"] = policy
    return load_config(cli_overrides=overrides)


def test_component_capability_metadata() -> None:
    assert CONTROLLER_REGISTRY["llm_controller"].requires_model is True
    assert SATISFACTION_REGISTRY["llm_satisfaction_updater"].requires_model is True
    assert REALIZER_REGISTRY["llm_user_realizer"].requires_model is True
    assert CONTROLLER_REGISTRY["mock_controller"].requires_model is False
    assert SATISFACTION_REGISTRY["mock_satisfaction_updater"].requires_model is False
    assert REALIZER_REGISTRY["mock_user_realizer"].requires_model is False


def test_mock_factory_loads_no_openrouter_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("mock mode must not load prompts, profiles, or OpenRouter clients")

    monkeypatch.setattr("user_simulator.factory.PromptTemplate.load", unexpected)
    monkeypatch.setattr("user_simulator.factory.load_model_profile", unexpected)
    monkeypatch.setattr("user_simulator.factory.OpenRouterStructuredClient", unexpected)
    config = _mock_config()
    components = build_simulator_components(
        config=config,
        environment=EnvironmentSettings(openrouter_api_key=""),
    )
    assert isinstance(components.controller, MockController)
    assert isinstance(components.satisfaction_updater, MockSatisfactionUpdater)
    assert isinstance(components.user_realizer, MockUserRealizer)
    assert components.prompts == {}
    assert components.model_profiles == {}
    assert components.uses_openrouter is False
    assert configured_components_require_openrouter(config) is False


def test_local_vllm_profiles_do_not_require_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        cli_overrides={
            "models": {
                key: "qwen_3_8_27b_vllm"
                for key in (
                    "controller",
                    "satisfaction",
                    "realizer_clear",
                    "realizer_abstract",
                )
            }
        }
    )
    monkeypatch.setattr(
        "user_simulator.factory.OpenRouterStructuredClient",
        lambda *args, **kwargs: object(),
    )

    assert configured_components_require_openrouter(config) is False
    components = build_simulator_components(
        config=config,
        environment=EnvironmentSettings(openrouter_api_key="", vllm_api_key=""),
    )
    assert components.uses_openrouter is False
    assert {profile.provider for profile in components.model_profiles.values()} == {"vllm"}


def test_factory_applies_fixed_medium_node_count() -> None:
    config = _mock_config(
        policy={
            "medium_min_nodes": 2,
            "medium_max_nodes": 2,
        }
    )
    components = build_simulator_components(config=config)

    selection = components.selection_policy.select(
        ["N1", "N2", "N3"], Difficulty.MEDIUM, random.Random(42)
    )
    assert selection.selected_nodes == ["N1", "N2"]


def test_config_rejects_inverted_medium_node_bounds() -> None:
    with pytest.raises(ValidationError, match="medium_max_nodes"):
        load_config(
            cli_overrides={
                "policy": {
                    "medium_min_nodes": 3,
                    "medium_max_nodes": 2,
                }
            }
        )


def test_registry_capability_does_not_depend_on_name_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        CONTROLLER_REGISTRY,
        "remote_controller",
        ComponentSpec(constructor=LLMController, requires_model=True),
    )
    config = load_config(
        cli_overrides={
            "components": {
                "controller": "remote_controller",
                "satisfaction_updater": "mock_satisfaction_updater",
                "user_realizer": "mock_user_realizer",
            }
        }
    )
    assert configured_components_require_openrouter(config) is True


def test_factory_rejects_unknown_component() -> None:
    config = load_config(cli_overrides={"components": {"controller": "does_not_exist"}})
    with pytest.raises(ConfigurationError, match="unknown controller"):
        build_simulator_components(config=config)


def test_demo_uses_factory_and_capability_metadata() -> None:
    source = inspect.getsource(demo.main)
    assert "build_simulator_components" in source
    assert "configured_components_require_openrouter" in source
    assert ".startswith(" not in source
    assert "LLMController(" not in source


def test_demo_cli_has_complete_policy_overrides() -> None:
    result = CliRunner().invoke(demo.app, ["--help"])
    assert result.exit_code == 0
    assert "--selection-policy" in result.output
    assert "--realization-policy" in result.output


def test_demo_reads_multiline_assistant_input(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = iter(["First paragraph.", "", "Second paragraph.", "/send"])
    monkeypatch.setattr(demo.console, "input", lambda *args, **kwargs: next(entered))

    assert demo._read_assistant_input() == "First paragraph.\n\nSecond paragraph."


def test_demo_executes_first_line_command_without_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = iter(["/audit", "not consumed"])
    monkeypatch.setattr(demo.console, "input", lambda *args, **kwargs: next(entered))

    assert demo._read_assistant_input() == "/audit"
    assert next(entered) == "not consumed"


def test_summary_audit_redacts_prompts_hidden_nodes_and_secrets(tmp_path: Path) -> None:
    logger = AuditLogger(
        "privacy",
        output_dir=tmp_path,
        run_id="summary",
        level="summary",
        config_snapshot={
            "system_prompt": "SYSTEM PROMPT SECRET",
            "openrouter_api_key": "key-secret",
            "headers": {"Authorization": "Bearer secret"},
        },
    )
    logger.log(
        "private",
        0,
        {
            "messages": [{"role": "system", "content": "SYSTEM PROMPT SECRET"}],
            "node_intent": "HIDDEN NODE INTENT",
            "selected_remaining_gaps": {"N2": "HIDDEN REMAINING GAP"},
            "raw_response": {
                "decisions": [
                    {
                        "node_id": "N2",
                        "exposable": True,
                        "reason": "HIDDEN NODE INTENT",
                    }
                ],
                "summary": "HIDDEN NODE INTENT",
            },
            "invalid_raw_response": {
                "updates": [
                    {
                        "node_id": "N2",
                        "status": "unsatisfied",
                        "reason": "INVALID RESPONSE SECRET",
                        "remaining_gap": "INVALID GAP SECRET",
                    }
                ]
            },
        },
    )
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in logger.run_dir.iterdir() if path.is_file()
    )
    for secret in (
        "SYSTEM PROMPT SECRET",
        "HIDDEN NODE INTENT",
        "HIDDEN REMAINING GAP",
        "INVALID RESPONSE SECRET",
        "INVALID GAP SECRET",
        "key-secret",
        "Bearer secret",
    ):
        assert secret not in text


@pytest.mark.parametrize(
    "fixture_path",
    [
        Path("tests/fixtures/prompt_cases.jsonl"),
        Path("tests/fixtures/review_regressions.jsonl"),
        Path("tests/fixtures/satisfaction_gap_regressions.jsonl"),
    ],
)
def test_contract_smoke_renders_real_inputs_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    fixture_path: Path,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    records = prompt_smoke_test._load_cases(fixture_path)
    report = prompt_smoke_test.run_contract_cases(records)
    assert report["mode"] == "contract"
    assert report["metrics"]["prompt_render_valid_rate"] == 1.0
    assert report["metrics"]["schema_fixture_valid_rate"] == 1.0
    assert report["metrics"]["semantic_fixture_valid_rate"] == 1.0
    assert "retry_rate" not in report["metrics"]
    assert all(case["rendered_messages"] for case in report["cases"])


def test_live_smoke_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        prompt_smoke_test,
        "EnvironmentSettings",
        lambda: EnvironmentSettings(openrouter_api_key=""),
    )
    result = CliRunner().invoke(
        prompt_smoke_test.app,
        ["--mode", "live", "--component", "controller"],
    )
    assert result.exit_code == 2
    assert "OPENROUTER_API_KEY is required" in result.output


def test_repository_hygiene() -> None:
    suffix = "_" + "v" + "2.yaml"
    obsolete_paths = [
        *(
            Path("configs/prompts") / f"{name}{suffix}"
            for name in (
                "controller",
                "satisfaction",
                "user_clear",
                "user_abstract",
            )
        ),
        Path("README_" + "v2.md"),
        Path("CHANGELOG_" + "V" + "2.md"),
        Path("src/user_simulator/llm") / "schemas",
    ]
    assert not [path for path in obsolete_paths if path.exists()]
    production_roots = [Path("src"), Path("configs")]
    class_pattern = re.compile(r"^class\s+\w+[V][123](?:\(|:)", re.MULTILINE)
    for root in production_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml"}:
                assert not class_pattern.search(path.read_text(encoding="utf-8")), path
