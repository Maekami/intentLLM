"""Replay exact saved Generator requests on local Qwen, changing only decoding mode.

No simulator, new planning, state commit, or retry. Each native event is flushed.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from assistant.config import load_model_profile
from assistant.goal_progression.contracts import validate_wire_required
from assistant.goal_progression.events import digest
from assistant.goal_progression.generator import validate_units
from assistant.goal_progression.runtime import public_usage
from assistant.goal_progression.schemas import GeneratorOutput, TurnContract
from assistant.llm.openrouter_client import OpenAICompatibleChatClient

ROOT = Path(__file__).resolve().parents[2]


async def probe(args):
    cases = [json.loads(line) for line in args.replay.read_text().splitlines() if line.strip()]
    if not 1 <= len(cases) <= 16:
        raise ValueError("Expected 1..16 saved single-turn replays")
    profile = load_model_profile(str(ROOT / "assistant/configs/models/qwen_3_6_27b_gp.yaml"))
    if profile.provider != "vllm":
        raise ValueError("This diagnostic is restricted to local vLLM")
    failed = False
    with args.output.open("x", encoding="utf-8") as handle:
        client = OpenAICompatibleChatClient(profile)
        try:
            for case in cases:
                request = next(
                    event["payload"]
                    for event in case["events"]
                    if event["event_type"] == "assistant_generator_requested"
                )
                generation = type(profile.generation["assistant"]).model_validate(
                    request["generation"]
                )
                base = {
                    "sample_id": case["sample_id"],
                    "turn": case["turn"],
                    "decoding": args.decoding,
                }

                def emit(kind, payload, base=base):
                    handle.write(
                        json.dumps(
                            {**base, "event_type": kind, "payload": payload}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    handle.flush()

                emit(
                    "assistant_generator_requested",
                    {
                        **request,
                        "structured_decoding": args.decoding,
                        "response_schema_sent": args.decoding == "schema",
                        "exact_messages_hash": digest(request["messages"]),
                    },
                )
                print(json.dumps({**base, "status": "requested"}), flush=True)
                started = time.perf_counter()
                try:
                    async with asyncio.timeout(
                        profile.goal_progression.recovery.call_timeout_seconds
                    ):
                        response = await client.generate(
                            messages=request["messages"],
                            generation=generation,
                            response_schema=request["response_schema"]
                            if args.decoding == "schema"
                            else None,
                        )
                    emit(
                        "assistant_generator_raw_result",
                        {
                            "raw_output": response.content,
                            "usage": public_usage(response.metadata),
                            "wall_seconds": time.perf_counter() - started,
                        },
                    )
                    data = json.loads(response.content)
                    validate_wire_required(data, request["response_schema"], "generator")
                    result = GeneratorOutput.model_validate(data)
                    contract = TurnContract.model_validate(
                        {
                            **request["context"]["turn_contract"],
                            "snapshot_version": 0,
                            "blocked_current": [],
                        }
                    )
                    validate_units(result, contract, {u.unit_id for u in contract.deliveries})
                    emit(
                        "assistant_generator_validated", {"structured_result": result.model_dump()}
                    )
                    print(
                        json.dumps(
                            {
                                **base,
                                "status": "contract_valid",
                                "issues": len(result.issues),
                                "body_characters": sum(len(u.text) for u in result.units),
                                "seconds": round(time.perf_counter() - started, 2),
                            }
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostic records errors, then fails
                    failed = True
                    emit(
                        "assistant_generator_failed",
                        {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "wall_seconds": time.perf_counter() - started,
                        },
                    )
                    print(
                        json.dumps({**base, "status": "failed", "error_type": type(exc).__name__}),
                        flush=True,
                    )
        finally:
            await client.client.close()
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New event JSONL; never overwrite"
    )
    parser.add_argument("--decoding", choices=("schema", "prompt"), required=True)
    raise SystemExit(asyncio.run(probe(parser.parse_args())))
