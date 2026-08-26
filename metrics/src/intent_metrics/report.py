from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from intent_metrics.evaluator import aggregate_metrics
from intent_metrics.models import EvaluationResult


def write_evaluation_outputs(
    output_dir: str | Path,
    result: EvaluationResult,
) -> tuple[Path, Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    episode_path = directory / "episode_metrics.jsonl"
    aggregate_path = directory / "aggregate_metrics.json"
    text_path = directory / "aggregate_metrics.txt"

    with episode_path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        for failure in result.failures:
            handle.write(
                json.dumps(
                    {"status": "evaluation_error", **failure.to_dict()},
                    ensure_ascii=False,
                )
                + "\n"
            )

    aggregate = aggregate_metrics(result.records, result.failures)
    aggregate_path.write_text(dumps_json_two_decimals(aggregate) + "\n", encoding="utf-8")
    text_path.write_text(render_aggregate_text(aggregate), encoding="utf-8")
    return episode_path, aggregate_path, text_path


def dumps_json_two_decimals(value: Any) -> str:
    """Serialize aggregate reports while retaining two decimal places for floats."""

    return _render_json(value, level=0)


def render_aggregate_text(report: dict[str, Any]) -> str:
    lines = [
        "Evaluation Metrics",
        f"Complete: {report['complete']}",
        (
            f"Episodes: {report['evaluated_episode_count']}/"
            f"{report['input_episode_count']} evaluated"
        ),
        "",
    ]
    models = report.get("models", {})
    for model, difficulties in models.items():
        lines.extend(
            [
                f"Model: {model}",
                (
                    "Difficulty | Avg. All-Node Exposure Turns ↓ | "
                    "Avg. All-Node Satisfaction Turns ↓ | Avg. Tokens ↓ | AITR ↑"
                ),
                "-" * 109,
            ]
        )
        for difficulty in ("easy", "medium", "hard"):
            if difficulty not in difficulties:
                continue
            metrics = difficulties[difficulty]
            lines.append(
                f"{difficulty.capitalize():10} | "
                f"{_format_metric(metrics['avg_all_node_exposure_turns']):31} | "
                f"{_format_metric(metrics['avg_all_node_satisfaction_turns']):35} | "
                f"{_format_metric(metrics['avg_tokens']):12} | "
                f"{_format_metric(metrics['aitr'])}"
            )
        lines.append("")
        for difficulty in ("easy", "medium", "hard"):
            if difficulty not in difficulties:
                continue
            diagnostics = difficulties[difficulty]["diagnostics"]
            lines.append(
                f"Diagnostics [{difficulty}]: episodes={diagnostics['episode_count']}, "
                f"exposure_failures={diagnostics['exposure_failures']}, "
                f"satisfaction_failures={diagnostics['satisfaction_failures']}, "
                f"aitr_api_failures={diagnostics['aitr_api_failures']}, "
                f"aitr_skipped={diagnostics['aitr_skipped']}, "
                f"warnings={diagnostics['warning_count']}"
            )
        lines.append("")

    errors = report.get("evaluation_errors", [])
    if errors:
        lines.append("Evaluation errors:")
        for error in errors:
            lines.append(
                f"- {error.get('run_dir')}: {error.get('error_type')}: {error.get('message')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _render_json(value: Any, *, level: int) -> str:
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("aggregate JSON cannot contain non-finite floats")
        return f"{value:.2f}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return "[]"
        rendered = [f"{child_indent}{_render_json(item, level=level + 1)}" for item in value]
        return "[\n" + ",\n".join(rendered) + f"\n{indent}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        rendered = []
        for key, item in value.items():
            rendered.append(
                f"{child_indent}{json.dumps(str(key), ensure_ascii=False)}: "
                f"{_render_json(item, level=level + 1)}"
            )
        return "{\n" + ",\n".join(rendered) + f"\n{indent}}}"
    raise TypeError(f"unsupported aggregate JSON value: {type(value).__name__}")
