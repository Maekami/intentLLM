"""Read-only GP iteration report renderer; never used as assistant input.

Print Markdown to stdout. The coding agent saves it through apply_patch after
adding the qualitative findings. No model requests or metric-source mutations.
"""

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import yaml
from intent_metrics.evaluator import evaluate_trace
from intent_metrics.traces import TraceLoader


def read_events(directory):
    path = directory / "events.jsonl"
    if not path.exists():
        return []
    records, seen = [], {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("payload", {})
            if payload.get("architecture_version") == "v2_contracts" and payload.get("event_id"):
                key = (payload.get("episode_id"), payload["event_id"])
                identity = (record["event_type"], payload)
                if key in seen:
                    if seen[key] != identity:
                        raise ValueError("Conflicting GP native event IDs")
                    continue
                seen[key] = identity
            records.append(record)
    return records


def changed_existing_fields(raw, parsed, path="$"):
    """Find canonical changes to supplied fields, ignoring newly filled defaults."""
    if isinstance(raw, dict) and isinstance(parsed, dict):
        return [
            changed
            for key, value in raw.items()
            for changed in (
                changed_existing_fields(value, parsed[key], path + "." + key)
                if key in parsed
                else [path + "." + key]
            )
        ]
    if isinstance(raw, list) and isinstance(parsed, list) and len(raw) == len(parsed):
        return [
            changed
            for index, (a, b) in enumerate(zip(raw, parsed))
            for changed in changed_existing_fields(a, b, f"{path}[{index}]")
        ]
    return [] if raw == parsed else [path]


async def report(root, revision):
    summaries = (
        [root / "batch_summary.json"]
        if (root / "batch_summary.json").exists()
        else list(root.glob("*/batch_summary.json"))
    )
    if len(summaries) != 1:
        raise ValueError(f"Expected one finished batch, got {len(summaries)}")
    batch = summaries[0].parent
    summary = json.loads(summaries[0].read_text())
    config = yaml.safe_load((batch / "batch_config.yaml").read_text())
    loader = TraceLoader()
    rows, records, errors, fallbacks, guards = [], [], [], [], []
    handoffs = []
    role_counts, actions, inter, assembled, shapes = (Counter() for _ in range(5))
    normalized = Counter()
    transport, unrecorded_transport, question_statuses = (Counter() for _ in range(3))
    target_words = []
    times, queues = [], defaultdict(list)
    clean = 0
    failed_generations = 0
    adjacent_states = 0
    versions, retry_codes, receipts = Counter(), Counter(), Counter()
    for run in summary["runs"]:
        directory = Path(run["run_dir"])
        if not directory.exists():
            directory = batch / directory.name
        events = read_events(directory)
        try:
            record = await evaluate_trace(loader.load(directory))
            records.append(record)
            e, s, tokens = (
                record.all_node_exposure_turn,
                record.all_node_satisfaction_turn,
                record.assistant_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - report unreadable episodes without hiding them
            e, s, tokens = None, None, None
            errors.append(f"{run['sample_id']}: metric load failed: {type(exc).__name__}: {exc}")
        rows.append((run["sample_id"], run["status"], run["turns"], e, s, tokens))
        for event in events:
            if event["event_type"] == "assistant_gp_contract_compiled":
                contract = event["payload"]["contract"]
                shapes[("v2_contract", len(contract["deliveries"]), len(contract["requests"]))] += 1
            if (
                event["event_type"] == "assistant_gp_state_committed"
                and event["payload"].get("architecture_version") != "v2_contracts"
            ):
                state = event["payload"].get("state") or {}
                adjacent_states += sum(g["scope"] == "adjacent" for g in state.get("goals", []))
            if event["event_type"] not in {
                "assistant_generation_completed",
                "assistant_generation_failed",
            }:
                continue
            delivered = event["event_type"] == "assistant_generation_completed"
            failed_generations += not delivered
            gp = (event["payload"].get("llm_call") or {}).get("goal_progression")
            if gp is None:
                errors.append(
                    f"{run['sample_id']} t{event['turn_index']}: GP call metadata unavailable"
                )
                continue
            if delivered:
                clean += not gp.get("fallbacks") and not gp.get("degradations")
                times.append(gp["wall_seconds"])
                receipt = gp.get("receipt") or {}
                receipts["delivered_units"] += len(receipt.get("delivered_units", []))
                receipts["issued_requests"] += len(receipt.get("issued_requests", []))
                if gp.get("architecture_version") == "v2_contracts":
                    state = gp.get("committed_state") or {}
                    adjacent_states += sum(g["scope"] == "adjacent" for g in state.get("goals", []))
            versions[gp.get("architecture_version", "v1_legacy")] += 1
            retry_codes.update(gp.get("retry_counts", {}))
            for degradation in gp.get("degradations", []):
                fallbacks.append(
                    f"{run['sample_id']} t{event['turn_index']}: optional prune {degradation}"
                )
            for call in gp["calls"]:
                role_counts[(call["role"], call["status"])] += 1
                attempts = (call.get("usage") or {}).get("transport_attempts")
                if attempts:
                    transport.update(
                        (
                            call["role"],
                            a.get("status", "unrecorded"),
                            a.get("finish_reason", "unrecorded"),
                        )
                        for a in attempts
                    )
                else:
                    unrecorded_transport[call["role"]] += 1
                if call["status"] == "validated" and "structured_result" in call:
                    try:
                        raw = json.loads(call["raw_output"])
                    except (ValueError, TypeError, KeyError):
                        errors.append(
                            f"{run['sample_id']} t{event['turn_index']}: validated raw JSON unavailable"
                        )
                    else:
                        if changed_existing_fields(raw, call["structured_result"]):
                            normalized[call["role"]] += 1
                queues[call["role"]].append(call.get("queue_seconds", 0))
                if call.get("validation_error"):
                    errors.append(
                        f"{run['sample_id']} t{event['turn_index']} {call['role']} attempt {call['attempt']}: {call['validation_error']}"
                    )
            if gp.get("fallbacks"):
                fallbacks.append(f"{run['sample_id']} t{event['turn_index']}: {gp['fallbacks']}")
            if gp.get("current_planning_handoff"):
                handoffs.append(
                    f"{run['sample_id']} t{event['turn_index']}: {gp['current_planning_handoff']}"
                )
            plan = gp.get("assembly") or {}
            if plan.get("repetition_guard_applied"):
                guards.append(f"{run['sample_id']} t{event['turn_index']}")
            intra_plan, inter_plan = gp.get("intra_plan"), gp.get("inter_plan")
            if gp.get("architecture_version") == "v2_contracts":
                for call in gp["calls"]:
                    if call["status"] != "validated":
                        continue
                    structured = call.get("structured_result", {})
                    if call["role"] == "intra":
                        intra_plan = structured.get("plan", structured)
                    if call["role"] == "inter":
                        inter_plan = structured.get("plan", structured)
                    if call["role"] in {"joint", "generator"} and "intra" in structured:
                        intra_plan = structured["intra"]
                        inter_plan = structured.get("inter", inter_plan)
            for item in (intra_plan or {}).get("items", []):
                actions[item["action"]] += 1
                if "deliveries" in item:
                    target_words.extend(len(d["target"].split()) for d in item["deliveries"])
                else:
                    question_statuses[
                        item.get("question_status")
                        or ("unassessed" if item.get("question") else "not_question")
                    ] += 1
                    target_words.append(len(item["action_target"].split()))
            if inter_plan:
                inter[inter_plan["action"]] += 1
            if plan:
                assembled[plan["optional_inter"]["action"]] += 1
                shapes[
                    (
                        plan["current_planning"],
                        bool(plan["selected_question"]),
                        plan["max_independent_questions"],
                    )
                ] += 1
    ready = len(records) == summary["sample_count"]
    e = mean(r.all_node_exposure_turn for r in records) if ready else None
    s = mean(r.all_node_satisfaction_turn for r in records) if ready else None
    token_ready = ready and all(r.assistant_tokens is not None for r in records)
    tokens = mean(r.assistant_tokens for r in records) if token_ready else None
    qualifies = (
        ready and summary["completed"] == summary["sample_count"] == 16 and e < 3.25 and s < 4.5
    )
    print(f"# {root.name}: GP revision {revision}\n")
    print(
        f"Batch: `{batch.name}`. Completed {summary['completed']}/{summary['sample_count']}; retries {summary['total_retries']}.\n"
    )
    print(f"All-node E: **{e}**; all-node S: **{s}**. Lower is better.")
    print(
        f"Episode-level visible tokens: **{tokens}**. Sum of delivered replies per episode, then mean; internal outputs excluded.\n"
    )
    if "v2_contracts" not in versions:
        print(
            f"Qualifies for the historical development streak (16/16, E<3.25, S<4.50): **{qualifies}**.\n"
        )
    else:
        print(
            "GP v2: no baseline-win verdict from historical development thresholds; use a frozen paired comparison.\n"
        )
    print(
        "Reference: historical Prompted Base on the same 16 samples. Simulator realizer prompts and routing differ from that historical run; this is not a controlled causal baseline comparison. No incomplete subset means are reported.\n"
    )
    print(
        "## Episodes\n\n| Sample | Status | Turns | E | S | Visible tokens |\n|---|---|---:|---:|---:|---:|"
    )
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    print(
        f"\n## Component execution\n\nDelivered turns: {len(times)}; clean turns: {clean}; mean GP wall seconds: {mean(times) if times else None}.\n"
    )
    print(
        f"Failed generation turns: {failed_generations}. Component calls below include delivered AND failed generations, including format retries; transport retries are counted separately. Failed outputs are not visible replies.\n"
    )
    print("| Role | Status | Component calls |\n|---|---|---:|")
    for (role, status), count in sorted(role_counts.items()):
        print(f"| {role} | {status} | {count} |")
    print(
        "\nRecorded transport attempts:\n\n| Role | Status | Finish reason | Attempts |\n|---|---|---|---:|"
    )
    for (role, status, finish), count in sorted(transport.items()):
        print(f"| {role} | {status} | {finish} | {count} |")
    print(
        f"\nComponent calls without transport-attempt records: `{dict(unrecorded_transport)}`. Unrecorded finish reasons are not inferred."
    )
    print(
        f"\nValidated calls with canonical field changes (ignoring filled defaults): `{dict(normalized)}`. These are not claimed as raw-contract-perfect model outputs."
    )
    print(
        f"\nIntra: `{dict(actions)}`; retained Inter plans (including fallback defaults): `{dict(inter)}`; assembled Inter: `{dict(assembled)}`."
    )
    print(
        f"\nTracker adjacent goal occurrences across states: {adjacent_states}. Question execution shapes: `{dict(shapes)}`."
    )
    print(
        f"\nIntra question-status labels: `{dict(question_statuses)}` (model estimates, not semantic ground truth)."
    )
    print(
        f"Intra validated target words (whitespace split): mean={mean(target_words) if target_words else None}, max={max(target_words) if target_words else None}; truncated/failed plans excluded from this length statistic."
    )
    print(f"\nMean per-role queue seconds: `{ {r: mean(q) for r, q in queues.items()} }`.")
    print(
        f"\nArchitecture versions: `{dict(versions)}`; independent retry counts: `{dict(retry_codes)}`."
    )
    print(
        f"Receipt facts: `{dict(receipts)}`; these certify text blocks, not semantic goal completion."
    )
    print(f"\nFallbacks: `{fallbacks}`. Repetition/stall guards: `{guards}`.\n")
    print(
        f"Empty-current-plan handoffs: `{handoffs}`. These are coverage handoffs, not proof of successful tracking.\n"
    )
    print("## Failures and validation errors\n")
    for row in summary["runs"]:
        if row.get("error"):
            print(f"- {row['sample_id']}: {row['error']}")
    for error in errors:
        print("- " + error.replace("\n", " "))
    if not errors and not summary["failed"]:
        print("None.")
    print("\n## Reproducibility\n")
    gp = config["assistant"]["goal_progression"]
    print(
        f"- GP recovery: {gp.get('recovery', gp.get('format_retries'))}; in-flight limit: {gp['max_in_flight_requests']}; timeout: {gp['turn_timeout_seconds']} seconds."
    )
    print(f"- Dataset SHA256: `{config['dataset']['sha256']}`.")
    generator_prompt = gp["prompts"].get("generator", {})
    output_format = gp.get("realization", {}).get(
        "output_format", generator_prompt.get("output_format", "text")
    )
    print(f"- Generator output format: `{output_format}`.")
    if gp.get("architecture_version") == "v2_contracts":
        print("- No routine Reviewer. Recovery only revisits existing producer nodes.")
    else:
        print(
            f"- Generator review enabled: `{bool(generator_prompt.get('review_operation'))}`; "
            f"review history: `{generator_prompt.get('review_history_format', 'compiled')}`."
        )
    for role, prompt in gp["prompts"].items():
        print(f"- {role} prompt: `{prompt['hash']}`.")
    for role, settings in gp.get("generation", {}).items():
        print(f"- {role} generation: `{json.dumps(settings, sort_keys=True)}`.")
    sources = sorted(Path("assistant/src/assistant/goal_progression").glob("*.py"))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path).encode() + b"\0" + path.read_bytes())
    print(
        "- GP source SHA256 at analysis (not proof of the source used by this batch): "
        f"`{digest.hexdigest()}`."
    )
    print(
        "- No simulator, pipeline, baseline, or metric implementation edits; no memory updates; sample-retries=0."
    )
    print(
        "- Only completed/retained attempts are analyzed; call errors are not hidden by episode reruns."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    asyncio.run(report(args.root, args.revision))
