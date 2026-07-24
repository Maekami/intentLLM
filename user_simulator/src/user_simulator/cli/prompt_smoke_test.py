import json
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from user_simulator.config import load_config
from user_simulator.engine.transitions import normalize_controller_result
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.llm.schema_registry import resolve_schema

app = typer.Typer(add_completion=False, help="Run offline V2 prompt-contract smoke cases.")
console = Console()


@app.command()
def main(
    component: str = typer.Option(
        "all",
        "--component",
        help="Component filter: all, controller, satisfaction, or user_generation.",
    ),
    cases: Path = typer.Option(
        Path("tests/fixtures/prompt_cases.jsonl"),
        "--cases",
        help="JSONL prompt smoke cases.",
    ),
    output_dir: Path = typer.Option(
        Path("runs/prompt_smoke"),
        "--output-dir",
        help="Directory for the JSON report.",
    ),
) -> None:
    allowed = {"all", "controller", "satisfaction", "user_generation"}
    if component not in allowed:
        raise typer.BadParameter(f"--component must be one of {sorted(allowed)}")
    records = _load_cases(cases)
    selected = [
        record for record in records if component == "all" or record["component"] == component
    ]
    if not selected:
        raise typer.BadParameter("no matching smoke cases")
    report = run_cases(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = Table(title="Prompt smoke report")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in report["metrics"].items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"Report: {destination}")


def run_cases(records: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = load_config()
    prompt_paths = {
        "controller": resolved.prompts.controller,
        "satisfaction": resolved.prompts.satisfaction,
        "user_generation": resolved.prompts.realizer_clear,
    }
    totals = {
        "schema_valid": 0,
        "semantic_valid": 0,
        "retried": 0,
        "prefix_violations": 0,
        "coverage_pass": 0,
        "unsupported_intent": 0,
    }
    labels = {
        "unsatisfied": 0,
        "partially_satisfied": 0,
        "satisfied": 0,
    }
    component_counts = {
        "controller": 0,
        "satisfaction": 0,
        "user_generation": 0,
    }
    details = []
    for record in records:
        component = record["component"]
        component_counts[component] += 1
        prompt = PromptTemplate.load(prompt_paths[component])
        schema = resolve_schema(prompt.schema_name, prompt.schema_version)
        response = record.get("input", {}).get("response", {})
        schema_valid = False
        semantic_valid = False
        error = None
        try:
            parsed = schema.model.model_validate_json(json.dumps(response))
            schema_valid = True
            totals["schema_valid"] += 1
            expected = record.get("expected_invariants", {})
            if component == "controller":
                required = expected.get("candidate_ids", [])
                normalized = normalize_controller_result(
                    parsed, required, expected.get("has_end_edge", False)
                )
                totals["prefix_violations"] += len(normalized.violations)
                semantic_valid = [item.node_id for item in parsed.decisions] == required
            elif component == "satisfaction":
                required = expected.get("exposed_ids", [])
                semantic_valid = [item.node_id for item in parsed.updates] == required
                for item in parsed.updates:
                    labels[item.status.value] += 1
            else:
                required = expected.get("selected_ids", [])
                coverage_ids = [item.node_id for item in parsed.coverage]
                semantic_valid = (
                    parsed.selected_node_ids == required
                    and coverage_ids == required
                    and all(item.covered for item in parsed.coverage)
                    and not parsed.contains_unsupported_intent
                )
                totals["coverage_pass"] += int(
                    coverage_ids == required and all(item.covered for item in parsed.coverage)
                )
                totals["unsupported_intent"] += int(parsed.contains_unsupported_intent)
            totals["semantic_valid"] += int(semantic_valid)
        except (ValidationError, ValueError) as exc:
            error = str(exc)
        details.append(
            {
                "case_id": record["case_id"],
                "component": component,
                "schema_valid": schema_valid,
                "semantic_valid": semantic_valid,
                "error": error,
            }
        )
    count = len(records)
    controller_count = component_counts["controller"] or 1
    user_count = component_counts["user_generation"] or 1
    metrics = {
        "cases": count,
        "schema_valid_rate": totals["schema_valid"] / count,
        "semantic_valid_rate": totals["semantic_valid"] / count,
        "retry_rate": totals["retried"] / count,
        "controller_prefix_violation_rate": (totals["prefix_violations"] / controller_count),
        "satisfaction_label_distribution": labels,
        "user_coverage_pass_rate": totals["coverage_pass"] / user_count,
        "unsupported_intent_self_check_rate": (totals["unsupported_intent"] / user_count),
    }
    return {"metrics": metrics, "cases": details}


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
            records.append(record)
    return records


if __name__ == "__main__":
    app()
