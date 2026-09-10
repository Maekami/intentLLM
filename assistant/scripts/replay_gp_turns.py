"""Explicit local-Qwen single-turn replay. Uses GP v2 receipts when available.

This command makes real model calls. --help and event restoration do not.
"""

import argparse
import asyncio
import json
from pathlib import Path

from assistant.config import load_model_profile
from assistant.domain.messages import ChatMessage
from assistant.goal_progression import GoalProgressionBaseline, GoalProgressionSession
from assistant.llm.openrouter_client import OpenAICompatibleChatClient

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "no_tracker", "no_intra", "no_inter", "joint", "no_anticipate")


def records(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sample_directory(run_dir, sample_id):
    summary = run_dir / "batch_summary.json"
    if summary.exists():
        runs = json.loads(summary.read_text(encoding="utf-8"))["runs"]
        run = next(item for item in runs if item["sample_id"] == sample_id)
        return run_dir / Path(run["run_dir"]).name
    # A batch may still be running. Resolve an exact sample prefix, without
    # manufacturing a completed summary or changing any of its original files.
    matches = [
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and path.name.rpartition("_")[0] == sample_id
        and (path / "events.jsonl").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one directory for sample {sample_id!r}; found {len(matches)}")
    return matches[0]


async def replay(run_dir, case, variant, on_event=None):
    sample_id, turn_text = case.rsplit(":", 1)
    turn = int(turn_text)
    if turn < 1:
        raise ValueError("turn must be positive")
    directory = sample_directory(run_dir, sample_id)
    events = records(directory / "events.jsonl")
    message = next(
        e["payload"]["user_message"]
        for e in events
        if e["event_type"] == "assistant_generation_requested" and e["turn_index"] == turn
    )
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in records(directory / "transcript.jsonl")
        if m["turn_index"] < turn - 1 or (m["turn_index"] == turn - 1 and m["role"] == "assistant")
    ]
    name = "qwen_3_6_27b_gp" + ("" if variant == "full" else "_" + variant)
    profile = load_model_profile(str(ROOT / "assistant/configs/models" / (name + ".yaml")))
    if profile.provider != "vllm":
        raise ValueError("this replay tool is restricted to local vLLM profiles")
    client = OpenAICompatibleChatClient(profile)
    session = GoalProgressionSession(
        client, profile.generation["assistant"], GoalProgressionBaseline(), profile=profile
    )
    session.replace_history(history)
    # Never pretend visible-only v1 history has exact v2 Need/request identities.
    prefix = [e for e in events if e["turn_index"] < turn]
    compatible = [
        e
        for e in prefix
        if e["event_type"] == "assistant_gp_state_committed"
        and e["payload"].get("architecture_version") == "v2_contracts"
        and e["payload"].get("variant") == variant
    ]
    restoration = "visible_only_unknown_request_history"
    if compatible:
        session.restore_events(prefix)
        restoration = "exact_v2_events"
    elif turn == 1:
        session.reset()
        restoration = "empty_episode"
    result = {
        "variant": variant,
        "sample_id": sample_id,
        "turn": turn,
        "restoration": restoration,
        "events": [],
    }

    def record_event(kind, payload):
        event = {"event_type": kind, "payload": payload}
        result["events"].append(event)
        if on_event is not None:
            on_event({"sample_id": sample_id, "turn": turn, "variant": variant, **event})

    try:
        result["reply"] = await session.respond(
            message,
            audit_sink=record_event,
        )
    except Exception as exc:  # noqa: BLE001 - preserve each diagnostic failure and its audit
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["metadata"] = session.last_call_metadata
        await client.client.close()
    return result


async def main(args):
    variants = args.variant or ["full"]
    if len(args.case) * len(variants) > 32:
        raise ValueError("at most 32 single-turn replays per invocation")
    failed = False
    # Exclusive creation protects previous verification artifacts.
    events_path = args.output.with_suffix(".events.jsonl")
    if events_path == args.output or args.output.exists() or events_path.exists():
        raise ValueError("Replay output and event sidecar must both be new, distinct files")
    with (
        args.output.open("x", encoding="utf-8") as handle,
        events_path.open("x", encoding="utf-8") as audit,
    ):

        def persist_event(event):
            audit.write(json.dumps(event, ensure_ascii=False) + "\n")
            audit.flush()
            kind, payload = event["event_type"], event["payload"]
            if kind.endswith(("_requested", "_validation_failed", "_validated")):
                print(
                    json.dumps(
                        {
                            "case": f"{event['sample_id']}:{event['turn']}",
                            "event": kind,
                            "attempt": payload.get("attempt"),
                            "code": payload.get("code"),
                        }
                    ),
                    flush=True,
                )

        for variant in variants:
            for case in args.case:
                result = await replay(args.run_dir, case, variant, persist_event)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                gp = result["metadata"].get("goal_progression", {})
                ok = gp.get("committed", False) and not gp.get("degradations")
                failed |= not ok
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "case": case,
                            "passed_without_fallback": ok,
                            "calls": [(c["role"], c["status"]) for c in gp.get("calls", [])],
                            "degradations": gp.get("degradations"),
                            "error": result.get("error"),
                            "restoration": result["restoration"],
                            "seconds": round(gp.get("wall_seconds", 0), 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="sample_id:assistant_turn")
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSONL file; never overwrite"
    )
    raise SystemExit(asyncio.run(main(parser.parse_args())))
