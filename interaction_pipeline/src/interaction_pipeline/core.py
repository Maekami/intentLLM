from __future__ import annotations

import json
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assistant.goal_progression import GoalProgressionSession
from assistant.session import AssistantSession
from user_simulator.audit.logger import AuditLogger
from user_simulator.engine.episode import Episode
from user_simulator.exceptions import EpisodeTurnLimitError

from interaction_pipeline.audit import render_human_audit

EventSink = Callable[[dict[str, Any]], Awaitable[None]]
TaskFinalizer = Callable[..., None]


@dataclass(frozen=True)
class RunResult:
    sample_id: str
    status: str
    turns: int
    run_dir: Path
    error: str | None = None


def record_setup_failure(
    *,
    sample_id: str,
    output_dir: str | Path,
    error: Exception,
    run_id: str | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> RunResult:
    resolved_run_id = run_id or f"{_safe_slug(sample_id)}_{uuid.uuid4().hex[:10]}"
    audit = AuditLogger(
        sample_id,
        output_dir=output_dir,
        run_id=resolved_run_id,
        level="full",
        enabled=True,
        config_snapshot=config_snapshot,
    )
    (audit.run_dir / "transcript.jsonl").touch()
    message = f"{type(error).__name__}: {error}"
    audit.log(
        "pipeline_setup_failed",
        0,
        {"error_type": type(error).__name__, "message": str(error)},
    )
    (audit.run_dir / "final_state.json").write_text(
        json.dumps(
            {"sample_id": sample_id, "status": "setup_failed", "error": message},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    render_human_audit(audit.run_dir)
    return RunResult(
        sample_id=sample_id,
        status="failed",
        turns=0,
        run_dir=audit.run_dir,
        error=message,
    )


async def run_interaction(
    *,
    episode: Episode,
    assistant: AssistantSession,
    audit: AuditLogger,
    event_sink: EventSink | None = None,
    update_memory: bool = True,
    task_finalizer: TaskFinalizer | None = None,
    stop_before_over_budget_generation: bool = False,
) -> RunResult:
    sample_id = episode.sample.sample_id
    status = "failed"
    error: str | None = None
    finalization_attempted = False
    audit.log(
        "pipeline_run_started",
        0,
        {
            "assistant_baseline": assistant.baseline.name,
            "assistant_model_profile": getattr(
                getattr(assistant.client, "profile", None), "profile_name", None
            ),
            "assistant_memory_framework": getattr(assistant, "memory_framework", None),
            "assistant_memory_updates_enabled": update_memory,
        },
    )
    await _emit(
        event_sink,
        "run_started",
        sample_id=sample_id,
        turn_index=0,
        run_dir=str(audit.run_dir),
    )
    try:
        first = await episode.start()
        current_user_message = first.user_message
        if current_user_message is None:
            raise RuntimeError("simulator returned no initial user message")
        await _emit_message(event_sink, sample_id, 0, "user", current_user_message)

        while not episode.state.terminated:
            if stop_before_over_budget_generation and episode.state.turn_index >= episode.max_turns:
                raise EpisodeTurnLimitError(
                    f"episode reached maximum of {episode.max_turns} assistant turns "
                    "without natural termination"
                )
            target_turn = episode.state.turn_index + 1
            audit.log(
                "assistant_generation_requested",
                target_turn,
                {
                    "baseline": assistant.baseline.name,
                    "user_message": current_user_message,
                },
            )
            try:
                if isinstance(assistant, GoalProgressionSession):
                    # Persist component results when they occur, even if a later
                    # component fails. Never publish private plans to event_sink/UI.
                    assistant_response = await assistant.respond(
                        current_user_message,
                        audit_sink=lambda kind, payload, turn=target_turn: audit.log(kind, turn, payload),
                    )
                else:
                    assistant_response = await assistant.respond(current_user_message)
            except Exception as exc:
                # Some baselines perform multiple private model calls before a
                # simulator-visible turn is accepted. Preserve their aggregate
                # metadata even when that local process ends in a typed failure.
                audit.log(
                    "assistant_generation_failed",
                    target_turn,
                    {
                        "baseline": assistant.baseline.name,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "llm_call": _public_llm_metadata(assistant.last_call_metadata),
                    },
                )
                raise
            audit.log(
                "assistant_generation_completed",
                target_turn,
                {
                    "baseline": assistant.baseline.name,
                    "assistant_message": assistant_response,
                    "llm_call": _public_llm_metadata(assistant.last_call_metadata),
                },
            )
            # Publish as soon as the complete assistant reply arrives. Simulator
            # evaluation can continue afterward without delaying the panel.
            await _emit_message(
                event_sink,
                sample_id,
                target_turn,
                "assistant",
                assistant_response,
            )
            turn = await episode.submit_assistant(assistant_response)
            if turn.terminal:
                break
            current_user_message = turn.user_message
            if current_user_message is None:
                raise RuntimeError("simulator returned no follow-up user message")
            await _emit_message(
                event_sink,
                sample_id,
                episode.state.turn_index,
                "user",
                current_user_message,
            )

        if update_memory:
            finalization_attempted = True
            (task_finalizer or _finalize_assistant_task)(
                assistant=assistant,
                audit=audit,
                task_id=sample_id,
                success=True,
                feedback=(
                    f"Episode terminated normally after {episode.state.turn_index} assistant turns."
                ),
                turn_index=episode.state.turn_index,
            )
        else:
            _log_memory_update_disabled(assistant, audit, episode.state.turn_index)
        status = "completed"
        audit.log(
            "pipeline_run_completed",
            episode.state.turn_index,
            {"turns": episode.state.turn_index},
        )
    except Exception as exc:  # noqa: BLE001 - sample failures must not abort a batch
        error = f"{type(exc).__name__}: {exc}"
        if update_memory and not finalization_attempted:
            finalization_attempted = True
            try:
                (task_finalizer or _finalize_assistant_task)(
                    assistant=assistant,
                    audit=audit,
                    task_id=sample_id,
                    success=False,
                    feedback=error,
                    turn_index=episode.state.turn_index,
                )
            except Exception as finalize_exc:  # noqa: BLE001 - preserve the primary error
                error += (
                    f"; memory finalization also failed: "
                    f"{type(finalize_exc).__name__}: {finalize_exc}"
                )
        audit.log(
            "pipeline_run_failed",
            episode.state.turn_index,
            {"error_type": type(exc).__name__, "message": str(exc)},
        )
    finally:
        audit.save_state(episode.state)
        render_human_audit(audit.run_dir)

    result = RunResult(
        sample_id=sample_id,
        status=status,
        turns=episode.state.turn_index,
        run_dir=audit.run_dir,
        error=error,
    )
    await _emit(
        event_sink,
        "run_completed" if status == "completed" else "run_failed",
        sample_id=sample_id,
        turn_index=episode.state.turn_index,
        status=status,
        error=error,
        run_dir=str(audit.run_dir),
    )
    return result


def _finalize_assistant_task(
    *,
    assistant: AssistantSession,
    audit: AuditLogger,
    task_id: str,
    success: bool,
    feedback: str,
    turn_index: int,
) -> None:
    try:
        result = assistant.finalize_task(
            task_id=task_id,
            success=success,
            feedback=feedback,
        )
    except Exception as exc:
        audit.log(
            "assistant_memory_update_failed",
            turn_index,
            {"error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
    if result is not None:
        event_type = (
            "assistant_memory_updated"
            if result.get("stored") is True
            else "assistant_memory_update_skipped"
        )
        audit.log(event_type, turn_index, result)


def _log_memory_update_disabled(
    assistant: AssistantSession,
    audit: AuditLogger,
    turn_index: int,
) -> None:
    framework = getattr(assistant, "memory_framework", None)
    if framework is None:
        return
    audit.log(
        "assistant_memory_update_disabled",
        turn_index,
        {
            "memory_framework": framework,
            "reason": "disabled_by_run_configuration",
        },
    )


def _public_llm_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # Visible messages are already recorded in transcript.jsonl. Hidden Qwen
    # thinking fields are deliberately not copied into the pipeline event log.
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"messages", "response", "reasoning_content", "reasoning_details"}
    }


async def _emit_message(
    sink: EventSink | None,
    sample_id: str,
    turn_index: int,
    role: str,
    content: str,
) -> None:
    await _emit(
        sink,
        "message",
        sample_id=sample_id,
        turn_index=turn_index,
        role=role,
        content=content,
    )


async def _emit(sink: EventSink | None, event_type: str, **values: Any) -> None:
    if sink is None:
        return
    await sink(
        {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            **values,
        }
    )


def _safe_slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (result or "sample")[:100]
