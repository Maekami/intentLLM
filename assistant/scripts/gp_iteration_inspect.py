"""Read-only development audit. Never imported by GP or supplied to model prompts."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from gp_analyze_iteration import read_events


def batch_dir(root):
    if (root / "batch_config.yaml").exists():
        return root
    candidates = list(root.glob("*/batch_config.yaml"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one batch below {root}, found {len(candidates)}")
    return candidates[0].parent


def fingerprint():
    paths = sorted(
        [*Path("assistant/src/assistant/goal_progression").glob("*.py")]
        + [*Path("assistant/configs/prompts").glob("gp_*.yaml")]
        + [*Path("assistant/configs/models").glob("qwen_3_6_27b_gp*.yaml")]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def completed_or_turn_limit(run):
    """Select by terminal outcome, never by turn count or metric availability."""
    if run["status"] == "completed":
        return True
    if run.get("outcome") is not None:
        return run["outcome"] == "FAILURE_TURN_LIMIT"
    # Older summaries may predate the explicit outcome field.
    return (run.get("error") or "").startswith("EpisodeTurnLimitError:")


async def inspect(root, *, sample=None, turns=False, progress=False, metric_population="all"):
    if metric_population not in {"all", "completed-or-turn-limit"}:
        raise ValueError(f"Unknown metric population: {metric_population}")
    batch = batch_dir(root)
    summary_path = batch / "batch_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    output = {"batch": str(batch), "finished": summary is not None}
    counts, retries, intra, inter, calls = (Counter() for _ in range(5))
    snapshots, adjacent = 0, 0
    rows = []
    directories = sorted(p for p in batch.iterdir() if (p / "events.jsonl").exists())
    for directory in directories:
        if sample and not directory.name.startswith(sample):
            continue
        events = read_events(directory)
        kinds = {e["event_type"] for e in events}
        counts["completed_samples"] += "pipeline_run_completed" in kinds
        counts["failed_samples"] += "pipeline_run_failed" in kinds
        if turns:
            transcript = directory / "transcript.jsonl"
            visible = [json.loads(line) for line in transcript.read_text().splitlines()]
            rows.append({"sample": directory.name, "transcript": visible})
            continue
        delivered = sum(e["event_type"] == "assistant_generation_completed" for e in events)
        failed = sum(e["event_type"] == "assistant_generation_failed" for e in events)
        counts["delivered_turns"] += delivered
        counts["failed_turns"] += failed
        counts["generator_skipped"] += sum(
            e["event_type"] == "assistant_generator_skipped" for e in events
        )
        for event in events:
            kind, payload = event["event_type"], event["payload"]
            if kind == "assistant_gp_snapshot_projected":
                snapshots += 1
                adjacent += sum(
                    g["scope"] == "adjacent" and g["status"] == "active"
                    for g in payload["snapshot"]["goals"]
                )
            if kind not in {"assistant_generation_completed", "assistant_generation_failed"}:
                continue
            gp = (payload.get("llm_call") or {}).get("goal_progression") or {}
            retries.update(gp.get("retry_counts", {}))
            for call in gp.get("calls", []):
                calls[f"{call['role']}:{call['status']}"] += 1
                if call["status"] != "validated":
                    continue
                result = call.get("structured_result") or {}
                if call["role"] == "intra":
                    intra.update(str(item["action"]) for item in result.get("items", []))
                if call["role"] == "inter":
                    inter[result.get("action", "unknown")] += 1
        if progress:
            rows.append(
                {
                    "sample": directory.name,
                    "turns": delivered,
                    "failed": failed,
                    "last_event": events[-1]["event_type"] if events else None,
                }
            )
    if not turns:
        output.update(counts)
        output.update(retries=retries, intra=intra, inter=inter, calls=calls)
        output.update(tracker_snapshots=snapshots, active_adjacent_occurrences=adjacent)
    if summary and not (turns or progress):
        from intent_metrics.evaluator import evaluate_trace
        from intent_metrics.traces import TraceLoader

        output.update(
            {
                k: summary[k]
                for k in (
                    "sample_count",
                    "completed",
                    "failed",
                    "infrastructure_failures",
                    "behavioral_failures",
                )
            }
        )
        for run in summary["runs"]:
            if sample and not run["sample_id"].startswith(sample):
                continue
            row = {k: run[k] for k in ("sample_id", "status", "turns", "error")}
            row["outcome"] = run.get("outcome")
            row["included_in_metrics"] = (
                metric_population == "all" or completed_or_turn_limit(run)
            )
            if not row["included_in_metrics"]:
                row.update(
                    E=None,
                    S=None,
                    visible_tokens=None,
                    metric_exclusion_reason="Neither completed nor turn-limit failure",
                )
                rows.append(row)
                continue
            try:
                record = await evaluate_trace(TraceLoader().load(batch / Path(run["run_dir"]).name))
                row.update(
                    E=record.all_node_exposure_turn,
                    S=record.all_node_satisfaction_turn,
                    visible_tokens=record.assistant_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - explicitly retain unreadable failures
                row.update(E=None, S=None, visible_tokens=None, metric_error=str(exc))
            rows.append(row)
        selected = [row for row in rows if row["included_in_metrics"]]
        output.update(
            metric_population=metric_population,
            metric_sample_count=len(selected),
            metric_excluded_sample_count=len(rows) - len(selected),
            metric_evaluation_failures=sum("metric_error" in row for row in selected),
        )
        for key in ("E", "S", "visible_tokens"):
            values = [r[key] for r in selected if r[key] is not None]
            output[key] = mean(values) if len(values) == len(selected) and selected else None
    output["samples"] = rows
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--sample")
    parser.add_argument("--turns", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument(
        "--metric-population",
        choices=("all", "completed-or-turn-limit"),
        default="all",
        help="Episode averaging population; excluded failures remain listed for audit.",
    )
    args = parser.parse_args()
    if args.fingerprint:
        print(fingerprint())
    else:
        if args.root is None:
            parser.error("root is required unless --fingerprint")
        print(
            json.dumps(
                asyncio.run(
                    inspect(
                        args.root,
                        sample=args.sample,
                        turns=args.turns,
                        progress=args.progress,
                        metric_population=args.metric_population,
                    )
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
