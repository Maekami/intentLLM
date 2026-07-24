import inspect
import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from user_simulator.cli import demo
from user_simulator.config import EnvironmentSettings, load_config, load_model_profile
from user_simulator.exceptions import ConfigurationError
from user_simulator.factory import build_simulator_components
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.llm.schema_registry import SCHEMA_REGISTRY, resolve_schema
from user_simulator.mock_components import (
    MockController,
    MockSatisfactionUpdater,
    MockUserRealizer,
)


@pytest.mark.parametrize(
    "path",
    [
        "configs/prompts/controller_v2.yaml",
        "configs/prompts/satisfaction_v2.yaml",
        "configs/prompts/user_clear_v2.yaml",
        "configs/prompts/user_abstract_v2.yaml",
    ],
)
def test_v2_prompt_contract(path: str) -> None:
    prompt = PromptTemplate.load(path)
    assert prompt.version == 2
    assert prompt.schema_version == 2
    assert prompt.schema is resolve_schema(prompt.schema_name, 2)


def test_prompt_hash_changes_with_rendered_content() -> None:
    prompt = PromptTemplate.load("configs/prompts/controller_v2.yaml")
    _, first = prompt.render(
        context="first",
        candidate_order='["N2"]',
        correction="",
    )
    _, second = prompt.render(
        context="second",
        candidate_order='["N2"]',
        correction="",
    )
    assert first["prompt_hash"] != second["prompt_hash"]
    assert first["schema_hash"] == prompt.schema.hash
    assert first["prompt_path"] == prompt.path


def test_critical_prompt_semantics_are_explicit() -> None:
    controller = PromptTemplate.load("configs/prompts/controller_v2.yaml").system
    satisfaction = PromptTemplate.load("configs/prompts/satisfaction_v2.yaml").system
    clear = PromptTemplate.load("configs/prompts/user_clear_v2.yaml").system
    abstract = PromptTemplate.load("configs/prompts/user_abstract_v2.yaml").system
    assert "Semantic relatedness alone is not enough" in controller
    assert "clarification question" in controller
    assert "user's own" in satisfaction
    assert "is not evidence that the intent has been satisfied" in " ".join(satisfaction.split())
    assert "partially_satisfied" in clear
    assert "Do not express any unselected" in clear
    assert "ABSTRACT DOES NOT MEAN UNRELATED" in abstract
    assert "diagnostic clue for every selected node" in abstract
    assert "initial turn" in clear.lower()
    assert "initial turn" in abstract.lower()


def test_unknown_prompt_placeholder_fails_at_load(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
name: bad
version: 2
schema_name: controller_result
schema_version: 2
system: test
user_template: "{context} {candidate_order} {correction} {unknown}"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown placeholders"):
        PromptTemplate.load(path)


def test_schema_registry_and_closed_schema() -> None:
    assert set(SCHEMA_REGISTRY) == {
        ("controller_result", 2),
        ("satisfaction_update_result", 2),
        ("user_generation_result", 2),
    }
    for spec in SCHEMA_REGISTRY.values():
        schema = spec.json_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]
        assert len(spec.hash) == 64


def test_user_schema_has_explicit_coverage_and_enum() -> None:
    schema = resolve_schema("user_generation_result", 2).json_schema
    assert "coverage" in schema["properties"]
    assert "coverage_check" not in schema["properties"]
    mode_ref = schema["properties"]["realization_mode"]["$ref"].split("/")[-1]
    assert schema["$defs"][mode_ref]["enum"] == ["clear", "abstract"]


def test_structured_settings_cannot_disable_v2_strictness() -> None:
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


def test_mock_component_factory_needs_no_api_key() -> None:
    config = load_config(
        cli_overrides={
            "components": {
                "controller": "mock_controller",
                "satisfaction_updater": "mock_satisfaction_updater",
                "user_realizer": "mock_user_realizer",
            }
        }
    )
    components = build_simulator_components(
        config=config,
        environment=EnvironmentSettings(openrouter_api_key=""),
    )
    assert isinstance(components.controller, MockController)
    assert isinstance(components.satisfaction_updater, MockSatisfactionUpdater)
    assert isinstance(components.user_realizer, MockUserRealizer)


def test_factory_rejects_unknown_component() -> None:
    config = load_config(cli_overrides={"components": {"controller": "does_not_exist"}})
    with pytest.raises(ConfigurationError, match="unknown controller"):
        build_simulator_components(config=config)


def test_prompt_override_is_resolved(tmp_path) -> None:
    source = PromptTemplate.load("configs/prompts/controller_v2.yaml")
    custom = tmp_path / "controller.yaml"
    custom.write_text(
        "\n".join(
            [
                "name: custom_controller",
                "version: 2",
                "schema_name: controller_result",
                "schema_version: 2",
                f"system: {json.dumps(source.system)}",
                f"user_template: {json.dumps(source.user_template)}",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(
        cli_overrides={
            "components": {
                "controller": "mock_controller",
                "satisfaction_updater": "mock_satisfaction_updater",
                "user_realizer": "mock_user_realizer",
            },
            "prompts": {"controller": str(custom)},
        }
    )
    components = build_simulator_components(config=config)
    assert components.prompts["controller"].name == "custom_controller"


def test_demo_uses_factory_instead_of_direct_llm_construction() -> None:
    source = inspect.getsource(demo.main)
    assert "build_simulator_components" in source
    assert "LLMController(" not in source


def test_demo_cli_prompt_override_and_mock_mode(tmp_path) -> None:
    source = PromptTemplate.load("configs/prompts/controller_v2.yaml")
    custom = tmp_path / "controller.yaml"
    custom.write_text(
        "\n".join(
            [
                "name: cli_override_controller",
                "version: 2",
                "schema_name: controller_result",
                "schema_version: 2",
                f"system: {json.dumps(source.system)}",
                f"user_template: {json.dumps(source.user_template)}",
            ]
        ),
        encoding="utf-8",
    )
    custom_config = tmp_path / "simulator.yaml"
    custom_config.write_text(
        f"audit:\n  output_dir: {json.dumps(str(tmp_path / 'runs'))}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        demo.app,
        [
            "--random-sample",
            "--mock",
            "--controller-prompt",
            str(custom),
            "--config",
            str(custom_config),
        ],
        input="/prompts\n/quit\n",
    )
    assert result.exit_code == 0, result.output
    assert "cli_override_controller" in result.output
    assert "Task summary:" not in result.output
