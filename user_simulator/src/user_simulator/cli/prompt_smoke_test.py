import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from user_simulator.config import (
    EnvironmentSettings,
    ModelProfile,
    SimulatorConfig,
    load_config,
    load_model_profile,
)
from user_simulator.controller.llm_controller import LLMController, controller_context
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.domain.enums import RealizationMode
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import (
    ControllerResult,
    SatisfactionUpdateResult,
    UserGenerationResult,
)
from user_simulator.domain.state import EpisodeState
from user_simulator.engine.transitions import normalize_controller_result
from user_simulator.llm.base import StructuredLLMClient
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.realizer.llm_realizer import (
    LLMUserRealizer,
    _generation_errors,
    realizer_context,
)
from user_simulator.satisfaction.llm_updater import (
    LLMSatisfactionUpdater,
    _specific_reason,
    satisfaction_context,
)

app = typer.Typer(add_completion=False, help="Run prompt contract or live smoke cases.")
console = Console()

_COMPONENTS = {"all", "controller", "satisfaction", "user_generation"}
_META_LANGUAGE = re.compile(
    r"\b(?:DAG|nodes?|selected intents?|satisfaction labels?|prompts?|"
    r"simulation|simulator|schemas?|structured output|hidden state)\b",
    re.IGNORECASE,
)
_CHARACTER_LANGUAGE = re.compile(
    r"\b(?:as an AI|I am an AI|language model|AI assistant|AI model)\b",
    re.IGNORECASE,
)
_MARKDOWN_LIST = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")


@app.command()
def main(
    mode: str = typer.Option(
        "contract",
        "--mode",
        help="Smoke mode: contract (offline) or live (OpenRouter).",
    ),
    component: str = typer.Option(
        "all",
        "--component",
        help="Component filter: all, controller, satisfaction, or user_generation.",
    ),
    cases: Path = typer.Option(
        Path("tests/fixtures/prompt_cases.jsonl"),
        "--cases",
        help="JSONL prompt smoke cases with real component inputs.",
    ),
    output_dir: Path = typer.Option(
        Path("runs/prompt_smoke"),
        "--output-dir",
        help="Directory for the prompt smoke JSON report.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Custom simulator YAML merged over defaults.",
    ),
) -> None:
    if mode not in {"contract", "live"}:
        raise typer.BadParameter("--mode must be contract or live")
    if component not in _COMPONENTS:
        raise typer.BadParameter(f"--component must be one of {sorted(_COMPONENTS)}")
    records = _load_cases(cases)
    selected = [
        record for record in records if component == "all" or record["component"] == component
    ]
    if not selected:
        raise typer.BadParameter("no matching smoke cases")

    resolved = load_config(config)
    if mode == "contract":
        report = run_contract_cases(selected, resolved)
    else:
        environment = EnvironmentSettings()
        if _records_require_openrouter(selected, resolved) and not environment.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY is required for live smoke mode.[/red]")
            raise typer.Exit(2)
        report = asyncio.run(run_live_cases(selected, resolved, environment))

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "report.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    table = Table(title=f"Prompt smoke report · {mode}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in report["metrics"].items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"Report: {destination}")


def run_contract_cases(
    records: list[dict[str, Any]],
    config: SimulatorConfig | None = None,
) -> dict[str, Any]:
    resolved = config or load_config()
    prompt_valid = 0
    fixture_count = 0
    schema_valid = 0
    semantic_valid = 0
    prefix_violations = 0
    details: list[dict[str, Any]] = []

    for record in records:
        component = record["component"]
        case: dict[str, Any] = {
            "case_id": record["case_id"],
            "component": component,
            "prompt_render_valid": False,
            "schema_fixture_valid": None,
            "semantic_fixture_valid": None,
            "error": None,
        }
        try:
            prompt, messages, metadata = _render_case(record, resolved)
            _validate_render(prompt, messages, metadata)
            prompt_valid += 1
            case["prompt_render_valid"] = True
            case["prompt_metadata"] = metadata
            case["rendered_messages"] = messages
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            case["error"] = f"prompt render failed: {exc}"
            details.append(case)
            continue

        fixture = record.get("fixture_response")
        if fixture is not None:
            fixture_count += 1
            try:
                parsed = prompt.schema.model.model_validate_json(json.dumps(fixture))
                schema_valid += 1
                case["schema_fixture_valid"] = True
                valid, violations = _semantic_fixture_result(record, parsed)
                prefix_violations += violations
                semantic_valid += int(valid)
                case["semantic_fixture_valid"] = valid
                case["fixture_response"] = parsed.model_dump(mode="json")
            except (ValidationError, ValueError) as exc:
                case["schema_fixture_valid"] = False
                case["semantic_fixture_valid"] = False
                case["error"] = str(exc)
        details.append(case)

    metrics = {
        "cases": len(records),
        "prompt_render_valid_rate": _rate(prompt_valid, len(records)),
        "schema_fixture_valid_rate": _rate(schema_valid, fixture_count),
        "semantic_fixture_valid_rate": _rate(semantic_valid, fixture_count),
        "prefix_normalization_violation_count": prefix_violations,
    }
    return {
        "mode": "contract",
        "metrics": metrics,
        "cases": details,
        "notes": [
            "Contract mode made no model request and reports no retry metric.",
            "User-generation coverage and unsupported-task-content fields are model self-reports.",
        ],
    }


async def run_live_cases(
    records: list[dict[str, Any]],
    config: SimulatorConfig | None = None,
    environment: EnvironmentSettings | None = None,
    *,
    client_factory: Callable[
        [ModelProfile, EnvironmentSettings], StructuredLLMClient
    ] = OpenRouterStructuredClient,
) -> dict[str, Any]:
    resolved = config or load_config()
    env = environment or EnvironmentSettings()
    totals = Counter()
    labels = Counter(
        {
            "unsatisfied": 0,
            "partially_satisfied": 0,
            "satisfied": 0,
        }
    )
    details: list[dict[str, Any]] = []
    user_messages: list[str] = []

    for record in records:
        component = record["component"]
        totals[f"component_{component}"] += 1
        case: dict[str, Any] = {
            "case_id": record["case_id"],
            "component": component,
            "structured_output_valid": False,
            "semantic_valid": False,
            "error": None,
        }
        component_instance: Any = None
        try:
            component_instance = _live_component(
                record,
                resolved,
                env,
                client_factory,
            )
            result = await _invoke_component(component_instance, record)
            case["structured_output_valid"] = True
            case["semantic_valid"] = True
            case["structured_response"] = result.model_dump(mode="json")
            totals["structured_valid"] += 1
            totals["semantic_valid"] += 1
            if component == "controller":
                required = _expected_ids(record, "candidate_ids", "candidates")
                normalized = normalize_controller_result(
                    result,
                    required,
                )
                if normalized.violations:
                    totals["prefix_violation_cases"] += 1
                case["normalization_violations"] = normalized.violations
            elif component == "satisfaction":
                for update in result.updates:
                    labels[update.status.value] += 1
            else:
                user_messages.append(result.user_message)
                coverage_pass = all(item.covered for item in result.coverage)
                totals["reported_coverage_pass"] += int(coverage_pass)
                totals["reported_unsupported_task_content"] += int(
                    result.contains_unsupported_task_content
                )
        except Exception as exc:  # noqa: BLE001 - live smoke isolates per-case failures
            case["error"] = str(exc)

        metadata = _component_metadata(component_instance)
        case["llm_call"] = metadata
        transport_retries = int(metadata.get("transport_retry_count") or 0)
        semantic_events = getattr(component_instance, "semantic_events", [])
        semantic_retries = max(0, len(semantic_events) - 1)
        totals["transport_retried"] += int(transport_retries > 0)
        totals["semantic_retried"] += int(semantic_retries > 0)
        case["transport_retry_count"] = transport_retries
        case["semantic_retry_count"] = semantic_retries
        details.append(case)

    diagnostics = _human_style_diagnostics(user_messages)
    user_count = totals["component_user_generation"]
    metrics = {
        "cases": len(records),
        "structured_output_valid_rate": _rate(totals["structured_valid"], len(records)),
        "semantic_valid_rate": _rate(totals["semantic_valid"], len(records)),
        "transport_retry_rate": _rate(totals["transport_retried"], len(records)),
        "semantic_retry_rate": _rate(totals["semantic_retried"], len(records)),
        "prefix_violation_rate": _rate(
            totals["prefix_violation_cases"], totals["component_controller"]
        ),
        "satisfaction_label_distribution": dict(labels),
        "reported_coverage_pass_rate": _rate(totals["reported_coverage_pass"], user_count),
        "reported_unsupported_task_content_rate": _rate(
            totals["reported_unsupported_task_content"], user_count
        ),
        **diagnostics,
    }
    return {
        "mode": "live",
        "metrics": metrics,
        "cases": details,
        "notes": [
            "Coverage and unsupported-task-content rates are model-reported checks, not independent proof.",
            "Human-style diagnostics are lightweight heuristics; blind human evaluation remains authoritative.",
        ],
    }


def _render_case(
    record: dict[str, Any],
    config: SimulatorConfig,
) -> tuple[PromptTemplate, list[dict[str, str]], dict[str, str]]:
    component = record["component"]
    values = record["input"]
    if component == "controller":
        prompt = PromptTemplate.load(config.prompts.controller)
        candidates = [DagNode.model_validate(item) for item in values["candidates"]]
        context = controller_context(
            history=_messages(values.get("history", [])),
            latest_assistant_response=str(values["latest_assistant_response"]),
            state=EpisodeState.model_validate(values["state"]),
            candidates=candidates,
        )
        order = [item.node_id for item in candidates]
        messages, metadata = prompt.render(
            context=context,
            candidate_order=json.dumps(order),
            correction="",
        )
    elif component == "satisfaction":
        prompt = PromptTemplate.load(config.prompts.satisfaction)
        exposed = [DagNode.model_validate(item) for item in values["exposed_nodes"]]
        context = satisfaction_context(
            history=_messages(values.get("history", [])),
            latest_assistant_response=str(values["latest_assistant_response"]),
            state=EpisodeState.model_validate(values["state"]),
            exposed_nodes=exposed,
        )
        order = [item.node_id for item in exposed]
        messages, metadata = prompt.render(
            context=context,
            exposed_node_order=json.dumps(order),
            correction="",
        )
    elif component == "user_generation":
        mode = RealizationMode(values["mode"])
        prompt_path = (
            config.prompts.realizer_clear
            if mode == RealizationMode.CLEAR
            else config.prompts.realizer_abstract
        )
        prompt = PromptTemplate.load(prompt_path)
        selected = [DagNode.model_validate(item) for item in values["selected_nodes"]]
        unselected = [
            DagNode.model_validate(item) for item in values.get("unselected_unresolved_nodes", [])
        ]
        sample = Sample.model_validate(values["sample"])
        context = realizer_context(
            selected_nodes=selected,
            unselected_unresolved_nodes=unselected,
            satisfaction=EpisodeState.model_validate(values["state"]).satisfaction,
            selected_remaining_gaps=dict(values.get("selected_remaining_gaps", {})),
            history=_messages(values.get("history", [])),
            latest_assistant_response=values.get("latest_assistant_response"),
            mode=mode,
            sample=sample,
        )
        order = [item.node_id for item in selected]
        messages, metadata = prompt.render(
            context=context,
            selected_node_order=json.dumps(order),
            correction="",
        )
    else:
        raise ValueError(f"unknown smoke component {component!r}")
    return prompt, messages, metadata


def _validate_render(
    prompt: PromptTemplate,
    messages: list[dict[str, str]],
    metadata: dict[str, str],
) -> None:
    if len(messages) != 2 or [message["role"] for message in messages] != ["system", "user"]:
        raise ValueError("rendered prompt must contain one system and one user message")
    if messages[0]["content"] != prompt.system:
        raise ValueError("rendered system message does not match configured prompt")
    expected = {"prompt_name", "prompt_path", "prompt_hash", "schema_name", "schema_hash"}
    if set(metadata) != expected:
        raise ValueError(f"rendered metadata keys differ: {sorted(set(metadata) ^ expected)}")
    if metadata["prompt_name"] != prompt.name or metadata["schema_name"] != prompt.schema_name:
        raise ValueError("rendered prompt metadata does not match configured prompt")
    if len(metadata["prompt_hash"]) != 64 or len(metadata["schema_hash"]) != 64:
        raise ValueError("rendered prompt or schema hash is malformed")


def _semantic_fixture_result(
    record: dict[str, Any],
    parsed: BaseModel,
) -> tuple[bool, int]:
    component = record["component"]
    if component == "controller":
        assert isinstance(parsed, ControllerResult)
        required = _expected_ids(record, "candidate_ids", "candidates")
        if [item.node_id for item in parsed.decisions] != required:
            return False, 0
        normalized = normalize_controller_result(
            parsed,
            required,
        )
        return True, len(normalized.violations)
    if component == "satisfaction":
        assert isinstance(parsed, SatisfactionUpdateResult)
        required = _expected_ids(record, "exposed_ids", "exposed_nodes")
        received = [item.node_id for item in parsed.updates]
        return (
            received == required
            and "END" not in received
            and len(received) == len(set(received))
            and all(_specific_reason(item.reason) for item in parsed.updates),
            0,
        )
    assert isinstance(parsed, UserGenerationResult)
    selected = _expected_ids(record, "selected_ids", "selected_nodes")
    mode = RealizationMode(record["input"]["mode"])
    return not _generation_errors(parsed, selected, mode), 0


def _live_component(
    record: dict[str, Any],
    config: SimulatorConfig,
    environment: EnvironmentSettings,
    client_factory: Callable[[ModelProfile, EnvironmentSettings], StructuredLLMClient],
) -> Any:
    component = record["component"]
    if component == "controller":
        profile = load_model_profile(config.models["controller"])
        client = client_factory(profile, environment)
        return LLMController(
            client,
            profile.generation["controller"],
            prompt=PromptTemplate.load(config.prompts.controller),
            model_profile_name=profile.profile_name,
            semantic_attempts=profile.retry.max_attempts,
        )
    if component == "satisfaction":
        profile = load_model_profile(config.models["satisfaction"])
        client = client_factory(profile, environment)
        return LLMSatisfactionUpdater(
            client,
            profile.generation["satisfaction"],
            prompt=PromptTemplate.load(config.prompts.satisfaction),
            model_profile_name=profile.profile_name,
            semantic_attempts=profile.retry.max_attempts,
        )
    mode = RealizationMode(record["input"]["mode"])
    profile_key = "realizer_clear" if mode == RealizationMode.CLEAR else "realizer_abstract"
    profile = load_model_profile(config.models[profile_key])
    client = client_factory(profile, environment)
    generation = profile.generation[profile_key]
    return LLMUserRealizer(
        client,
        generation,
        generation,
        clear_prompt=PromptTemplate.load(config.prompts.realizer_clear),
        abstract_prompt=PromptTemplate.load(config.prompts.realizer_abstract),
        model_profile_name=profile.profile_name,
        abstract_client=client,
        abstract_model_profile_name=profile.profile_name,
        semantic_attempts=profile.retry.max_attempts,
        abstract_semantic_attempts=profile.retry.max_attempts,
    )


def _records_require_openrouter(
    records: list[dict[str, Any]],
    config: SimulatorConfig,
) -> bool:
    profile_keys: set[str] = set()
    for record in records:
        component = record["component"]
        if component == "controller":
            profile_keys.add("controller")
        elif component == "satisfaction":
            profile_keys.add("satisfaction")
        else:
            mode = RealizationMode(record["input"]["mode"])
            profile_keys.add(
                "realizer_clear" if mode == RealizationMode.CLEAR else "realizer_abstract"
            )
    return any(
        load_model_profile(config.models[key]).provider == "openrouter"
        for key in profile_keys
    )


async def _invoke_component(component: Any, record: dict[str, Any]) -> BaseModel:
    values = record["input"]
    if record["component"] == "controller":
        return await component.decide(
            history=_messages(values.get("history", [])),
            latest_assistant_response=values["latest_assistant_response"],
            state=EpisodeState.model_validate(values["state"]),
            candidates=[DagNode.model_validate(item) for item in values["candidates"]],
        )
    if record["component"] == "satisfaction":
        return await component.update(
            history=_messages(values.get("history", [])),
            latest_assistant_response=values["latest_assistant_response"],
            state=EpisodeState.model_validate(values["state"]),
            exposed_nodes=[DagNode.model_validate(item) for item in values["exposed_nodes"]],
        )
    state = EpisodeState.model_validate(values["state"])
    return await component.generate(
        selected_nodes=[DagNode.model_validate(item) for item in values["selected_nodes"]],
        unselected_unresolved_nodes=[
            DagNode.model_validate(item) for item in values.get("unselected_unresolved_nodes", [])
        ],
        satisfaction=state.satisfaction,
        selected_remaining_gaps=dict(values.get("selected_remaining_gaps", {})),
        history=_messages(values.get("history", [])),
        latest_assistant_response=values.get("latest_assistant_response"),
        mode=RealizationMode(values["mode"]),
        sample=Sample.model_validate(values["sample"]),
    )


def _component_metadata(component: Any) -> dict[str, Any]:
    if component is None:
        return {}
    direct = getattr(component, "last_call_metadata", None)
    if isinstance(direct, dict) and direct:
        return dict(direct)
    client = getattr(component, "client", None)
    metadata = getattr(client, "last_call_metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _human_style_diagnostics(messages: list[str]) -> dict[str, float | None]:
    if not messages:
        return {
            "user_character_violation_rate": None,
            "meta_language_violation_rate": None,
            "over_elaboration_rate": None,
            "repeated_template_opening_rate": None,
        }
    openings = [" ".join(re.findall(r"\w+", message.lower())[:4]) for message in messages]
    opening_counts = Counter(opening for opening in openings if opening)
    repeated = sum(1 for opening in openings if opening and opening_counts[opening] > 1)
    return {
        "user_character_violation_rate": _rate(
            sum(bool(_CHARACTER_LANGUAGE.search(message)) for message in messages),
            len(messages),
        ),
        "meta_language_violation_rate": _rate(
            sum(bool(_META_LANGUAGE.search(message)) for message in messages),
            len(messages),
        ),
        "over_elaboration_rate": _rate(
            sum(
                len(message) > 600
                or message.count("\n") > 5
                or bool(_MARKDOWN_LIST.search(message))
                for message in messages
            ),
            len(messages),
        ),
        "repeated_template_opening_rate": _rate(repeated, len(messages)),
    }


def _messages(values: list[dict[str, Any]]) -> list[ChatMessage]:
    return [ChatMessage.model_validate(item) for item in values]


def _expected_ids(
    record: dict[str, Any],
    invariant_key: str,
    input_key: str,
) -> list[str]:
    expected = record.get("expected_invariants", {}).get(invariant_key)
    if expected is not None:
        return list(expected)
    return [item["node_id"] for item in record["input"][input_key]]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _load_cases(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"case_id", "component", "input", "expected_invariants"}
            missing = required - set(record)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields {sorted(missing)}")
            if record["component"] not in _COMPONENTS - {"all"}:
                raise ValueError(f"{path}:{line_number} unknown component {record['component']!r}")
            records.append(record)
    return records


if __name__ == "__main__":
    app()
