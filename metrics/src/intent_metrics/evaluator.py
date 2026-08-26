from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean

from intent_metrics.aitr import AITRJudge, AITRJudgment, score_to_aitr
from intent_metrics.cache import AITRCache, AITRCacheRecord, conversation_hash
from intent_metrics.deterministic import (
    FAILURE_TURN,
    compute_assistant_tokens,
    compute_dag_turn_metrics,
)
from intent_metrics.models import (
    EpisodeMetricRecord,
    EpisodeTrace,
    EvaluationFailure,
    EvaluationResult,
)
from intent_metrics.prompt import AITR_PROMPT_VERSION
from intent_metrics.traces import TraceLoader


async def evaluate_trace(
    trace: EpisodeTrace,
    *,
    judge: AITRJudge | None = None,
    cache: AITRCache | None = None,
) -> EpisodeMetricRecord:
    dag = compute_dag_turn_metrics(trace)
    tokens = compute_assistant_tokens(trace)
    warnings = list(dag.warnings)
    if tokens.fallback_turns:
        warnings.append(
            "assistant output_tokens unavailable; used answer/thinking fallback on turns "
            f"{list(tokens.fallback_turns)}"
        )

    base = {
        "episode_id": trace.episode_id,
        "sample_id": trace.sample_id,
        "run_dir": str(trace.run_dir),
        "difficulty": trace.difficulty,
        "assistant_model": trace.assistant_model,
        "source_outcome": trace.source_outcome,
        "all_node_exposure_turn": dag.all_node_exposure_turn,
        "all_node_satisfaction_turn": dag.all_node_satisfaction_turn,
        "assistant_tokens": tokens.assistant_tokens,
        "warnings": tuple(warnings),
    }
    if judge is None:
        return EpisodeMetricRecord(**base)

    base.update(
        {
            "aitr_judge_config_hash": judge.config_hash,
            "aitr_prompt_version": judge.prompt_version,
            "aitr_prompt_hash": judge.prompt_hash,
        }
    )

    digest = conversation_hash(trace.conversation)
    cached = (
        cache.get(
            episode_id=trace.episode_id,
            conversation_hash=digest,
            judge_model=judge.model_id,
            reasoning_effort=judge.reasoning_effort,
            judge_config_hash=judge.config_hash,
            prompt_version=judge.prompt_version,
            prompt_hash=judge.prompt_hash,
        )
        if cache is not None
        else None
    )
    if cached is not None:
        return EpisodeMetricRecord(
            **base,
            aitr_raw_score=cached.raw_score,
            aitr_normalized_score=cached.normalized_score,
            aitr=cached.aitr,
            aitr_reason=cached.reason,
            aitr_cached=True,
        )

    try:
        response = await judge.judge(trace.conversation)
        judgment = AITRJudgment.model_validate(response.judgment)
        normalized, aitr = score_to_aitr(judgment.score)
        if cache is not None:
            cache.put(
                AITRCacheRecord(
                    episode_id=trace.episode_id,
                    conversation_hash=digest,
                    judge_model=judge.model_id,
                    reasoning_effort=judge.reasoning_effort,
                    judge_config_hash=judge.config_hash,
                    prompt_version=judge.prompt_version,
                    prompt_hash=judge.prompt_hash,
                    raw_judge_response=response.raw_response,
                    raw_score=judgment.score,
                    normalized_score=normalized,
                    aitr=aitr,
                    reason=judgment.reason,
                    judge_metadata=response.metadata,
                )
            )
        return EpisodeMetricRecord(
            **base,
            aitr_raw_score=judgment.score,
            aitr_normalized_score=normalized,
            aitr=aitr,
            aitr_reason=judgment.reason,
        )
    except Exception as exc:  # noqa: BLE001 - judge failures remain explicit episode results
        return EpisodeMetricRecord(
            **base,
            aitr_error=f"{type(exc).__name__}: {exc}",
        )


async def evaluate_runs(
    run_dirs: Iterable[str],
    *,
    loader: TraceLoader,
    judge: AITRJudge | None = None,
    cache: AITRCache | None = None,
) -> EvaluationResult:
    result = EvaluationResult()
    for raw_run_dir in run_dirs:
        try:
            trace = loader.load(raw_run_dir)
            result.records.append(await evaluate_trace(trace, judge=judge, cache=cache))
        except Exception as exc:  # noqa: BLE001 - isolate and retain each trace failure
            result.failures.append(
                EvaluationFailure(
                    run_dir=str(raw_run_dir),
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
    return result


def aggregate_metrics(
    records: Iterable[EpisodeMetricRecord],
    failures: Iterable[EvaluationFailure] = (),
) -> dict:
    episode_records = list(records)
    evaluation_failures = list(failures)
    grouped: dict[tuple[str, str], list[EpisodeMetricRecord]] = defaultdict(list)
    for record in episode_records:
        grouped[(record.assistant_model, record.difficulty)].append(record)

    models: dict[str, dict[str, dict]] = {}
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    for (model, difficulty), items in sorted(
        grouped.items(), key=lambda item: (item[0][0], difficulty_order[item[0][1]])
    ):
        aitr_ready = all(item.aitr is not None and item.aitr_error is None for item in items)
        models.setdefault(model, {})[difficulty] = {
            "avg_all_node_exposure_turns": round(
                fmean(item.all_node_exposure_turn for item in items), 2
            ),
            "avg_all_node_satisfaction_turns": round(
                fmean(item.all_node_satisfaction_turn for item in items), 2
            ),
            "avg_tokens": round(fmean(item.assistant_tokens for item in items), 2),
            "aitr": round(fmean(item.aitr for item in items), 2) if aitr_ready else None,
            "diagnostics": {
                "episode_count": len(items),
                "exposure_failures": sum(
                    item.all_node_exposure_turn == FAILURE_TURN for item in items
                ),
                "satisfaction_failures": sum(
                    item.all_node_satisfaction_turn == FAILURE_TURN for item in items
                ),
                "aitr_api_failures": sum(item.aitr_error is not None for item in items),
                "aitr_skipped": sum(
                    item.aitr is None and item.aitr_error is None for item in items
                ),
                "warning_count": sum(len(item.warnings) for item in items),
            },
            "status": "complete" if aitr_ready else "aitr_incomplete",
        }

    all_aitr_complete = all(
        record.aitr is not None and not record.aitr_error for record in episode_records
    )
    complete = bool(episode_records) and not evaluation_failures and all_aitr_complete
    prompt_versions = {
        record.aitr_prompt_version
        for record in episode_records
        if record.aitr_prompt_version is not None
    }
    reported_prompt_version = (
        next(iter(prompt_versions))
        if len(prompt_versions) == 1
        else (AITR_PROMPT_VERSION if not prompt_versions else None)
    )
    return {
        "prompt_version": reported_prompt_version,
        "complete": complete,
        "input_episode_count": len(episode_records) + len(evaluation_failures),
        "evaluated_episode_count": len(episode_records),
        "models": models,
        "evaluation_errors": [item.to_dict() for item in evaluation_failures],
    }
