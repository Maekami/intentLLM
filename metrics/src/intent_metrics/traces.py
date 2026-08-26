from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from user_simulator.data.loader import DatasetLoader

from intent_metrics.deterministic import T_MAX
from intent_metrics.errors import TraceLoadError
from intent_metrics.models import ConversationMessage, DifficultyName, EpisodeTrace

_DIFFICULTIES = {"easy", "medium", "hard"}


class TraceLoader:
    """Load canonical metric inputs from one interaction-pipeline run directory."""

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        self.dataset_override = (
            Path(dataset_path).expanduser().resolve() if dataset_path is not None else None
        )
        self._datasets: dict[Path, DatasetLoader] = {}

    def load(self, run_dir: str | Path) -> EpisodeTrace:
        directory = Path(run_dir).expanduser().resolve()
        if not directory.is_dir():
            raise TraceLoadError(f"run directory does not exist: {directory}")

        events = tuple(_read_jsonl(directory / "events.jsonl"))
        if not events:
            raise TraceLoadError(f"run contains no audit events: {directory}")
        snapshot = _read_yaml(directory / "config_snapshot.yaml")
        final_state = _read_json(directory / "final_state.json", required=False)

        sample_id = _resolve_sample_id(events, final_state, snapshot)
        difficulty = _resolve_difficulty(events, final_state, snapshot)
        dataset_path = self.dataset_override or _resolve_dataset_path(directory, snapshot)
        loader = self._datasets.setdefault(dataset_path, DatasetLoader(dataset_path))
        try:
            sample = loader.load_sample_by_id(sample_id)
        except Exception as exc:  # canonical loader supplies the useful source detail
            raise TraceLoadError(
                f"cannot load sample {sample_id!r} from {dataset_path}: {exc}"
            ) from exc

        intent_node_ids = tuple(
            node.node_id for node in sample.reason_dag.nodes if node.node_type == "intent"
        )
        if not intent_node_ids:
            raise TraceLoadError(f"sample {sample_id!r} has no non-terminal intent nodes")

        initial_exposed = _resolve_initial_exposed(events, intent_node_ids)
        source_outcome = _resolve_source_outcome(events)
        conversation = _load_conversation(directory / "transcript.jsonl")
        if not any(message.role == "assistant" for message in conversation):
            raise TraceLoadError(f"run contains no accepted assistant response: {directory}")

        return EpisodeTrace(
            run_dir=directory,
            run_id=directory.name,
            sample_id=sample_id,
            difficulty=difficulty,
            assistant_model=_resolve_assistant_model(events, snapshot),
            intent_node_ids=intent_node_ids,
            initial_exposed_node_ids=initial_exposed,
            events=events,
            conversation=conversation,
            source_outcome=source_outcome,
        )


def discover_run_dirs(path: str | Path) -> list[Path]:
    """Discover run directories from either a run or a pipeline batch directory."""

    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise TraceLoadError(f"input directory does not exist: {source}")
    if (source / "events.jsonl").is_file():
        return [source]

    summary_path = source / "batch_summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        raw_runs = summary.get("runs")
        if not isinstance(raw_runs, list):
            raise TraceLoadError(f"{summary_path} has no valid runs list")
        discovered: list[Path] = []
        for index, item in enumerate(raw_runs):
            if not isinstance(item, dict) or not isinstance(item.get("run_dir"), str):
                raise TraceLoadError(f"{summary_path}: runs[{index}] has no run_dir")
            candidate = Path(item["run_dir"]).expanduser()
            if not candidate.is_absolute():
                candidate = source / candidate
            candidate = candidate.resolve()
            if not candidate.is_dir():
                relocated = source / candidate.name
                if relocated.is_dir():
                    candidate = relocated.resolve()
            discovered.append(candidate)
        return _deduplicate_paths(discovered)

    discovered = sorted(item.parent.resolve() for item in source.rglob("events.jsonl"))
    if not discovered:
        raise TraceLoadError(f"no interaction runs found under {source}")
    return _deduplicate_paths(discovered)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TraceLoadError(f"required trace file is missing: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TraceLoadError(f"{path}:{line_number} must contain a JSON object")
                records.append(value)
    except json.JSONDecodeError as exc:
        raise TraceLoadError(f"{path}:{exc.lineno}: invalid JSON: {exc.msg}") from exc
    except OSError as exc:
        raise TraceLoadError(f"cannot read {path}: {exc}") from exc
    return records


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise TraceLoadError(f"required JSON file is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceLoadError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceLoadError(f"{path} must contain a JSON object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TraceLoadError(f"required config snapshot is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TraceLoadError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceLoadError(f"{path} must contain a YAML mapping")
    return value


def _resolve_sample_id(
    events: tuple[dict[str, Any], ...],
    final_state: dict[str, Any],
    snapshot: dict[str, Any],
) -> str:
    candidates: list[str] = []
    for value in (final_state.get("sample_id"), snapshot.get("sample_id")):
        if isinstance(value, str) and value:
            candidates.append(value)
    for event in events:
        value = event.get("sample_id")
        if isinstance(value, str) and value:
            candidates.append(value)
            break
    if not candidates:
        raise TraceLoadError("sample_id is absent from state, config snapshot, and events")
    if len(set(candidates)) != 1:
        raise TraceLoadError(f"inconsistent sample_id values in trace: {sorted(set(candidates))}")
    return candidates[0]


def _resolve_difficulty(
    events: tuple[dict[str, Any], ...],
    final_state: dict[str, Any],
    snapshot: dict[str, Any],
) -> DifficultyName:
    candidates: list[str] = []
    state_value = final_state.get("difficulty")
    if isinstance(state_value, str):
        candidates.append(state_value)
    pipeline = snapshot.get("pipeline")
    if isinstance(pipeline, dict) and isinstance(pipeline.get("run"), dict):
        value = pipeline["run"].get("difficulty")
        if isinstance(value, str):
            candidates.append(value)
    for event in events:
        if event.get("event_type") != "episode_started":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("difficulty"), str):
            candidates.append(payload["difficulty"])
        break
    invalid = sorted({item for item in candidates if item not in _DIFFICULTIES})
    if invalid:
        raise TraceLoadError(f"invalid difficulty values in trace: {invalid}")
    if not candidates:
        raise TraceLoadError("difficulty is absent from state, config snapshot, and events")
    if len(set(candidates)) != 1:
        raise TraceLoadError(f"inconsistent difficulty values in trace: {sorted(set(candidates))}")
    return candidates[0]  # type: ignore[return-value]


def _resolve_dataset_path(run_dir: Path, snapshot: dict[str, Any]) -> Path:
    simulator = snapshot.get("simulator")
    raw = simulator.get("dataset_path") if isinstance(simulator, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        raise TraceLoadError(
            "config snapshot has no simulator.dataset_path; pass an explicit dataset path"
        )
    value = Path(raw).expanduser()
    if value.is_absolute():
        if value.is_file():
            return value.resolve()
        raise TraceLoadError(
            f"snapshot dataset no longer exists: {value}; pass an explicit dataset path"
        )

    candidates = [Path.cwd() / value, run_dir / value]
    pipeline = snapshot.get("pipeline")
    projects = pipeline.get("projects") if isinstance(pipeline, dict) else None
    simulator_root = projects.get("simulator_root") if isinstance(projects, dict) else None
    if isinstance(simulator_root, str):
        root_value = Path(simulator_root).expanduser()
        if root_value.is_absolute():
            candidates.append(root_value / value)
        else:
            candidates.extend(ancestor / root_value / value for ancestor in run_dir.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(item) for item in candidates[:5])
    raise TraceLoadError(
        f"cannot resolve snapshot dataset path {raw!r} (tried {tried}); "
        "pass an explicit dataset path"
    )


def _resolve_initial_exposed(
    events: tuple[dict[str, Any], ...], intent_node_ids: tuple[str, ...]
) -> tuple[str, ...]:
    for event in events:
        if event.get("event_type") != "episode_started":
            continue
        payload = event.get("payload")
        frontier = payload.get("initial_frontier") if isinstance(payload, dict) else None
        if not isinstance(frontier, str) or frontier not in intent_node_ids:
            raise TraceLoadError(f"invalid episode_started.initial_frontier: {frontier!r}")
        return (frontier,)
    raise TraceLoadError("episode_started event is missing; initial exposure state is unknown")


def _resolve_source_outcome(events: tuple[dict[str, Any], ...]) -> str:
    outcomes = [
        event
        for event in events
        if event.get("event_type") in {"pipeline_run_completed", "pipeline_run_failed"}
    ]
    if not outcomes:
        raise TraceLoadError("pipeline run has no completion/failure event")
    outcome = outcomes[-1]
    turn = outcome.get("turn_index")
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
        raise TraceLoadError(f"pipeline outcome has invalid turn_index: {turn!r}")
    if outcome.get("event_type") == "pipeline_run_completed":
        return "completed" if turn <= T_MAX else "budget_exhausted"
    payload = outcome.get("payload")
    error_type = payload.get("error_type") if isinstance(payload, dict) else None
    if error_type == "EpisodeTurnLimitError" and turn >= T_MAX:
        return "budget_exhausted"
    if error_type == "EpisodeTurnLimitError":
        raise TraceLoadError(
            f"episode stopped at turn {turn}, before the {T_MAX}-turn evaluation budget"
        )
    raise TraceLoadError(f"pipeline failed for a non-evaluation reason: {error_type or 'unknown'}")


def _load_conversation(path: Path) -> tuple[ConversationMessage, ...]:
    records = _read_jsonl(path)
    messages: list[ConversationMessage] = []
    for index, record in enumerate(records, 1):
        role = record.get("role")
        if role not in {"user", "assistant"}:
            continue
        turn = record.get("turn_index")
        content = record.get("content")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
            raise TraceLoadError(f"{path}:{index} has invalid turn_index {turn!r}")
        if turn > T_MAX:
            continue
        if not isinstance(content, str) or not content.strip():
            raise TraceLoadError(f"{path}:{index} has empty message content")
        messages.append(ConversationMessage(turn_index=turn, role=role, content=content))
    # A budget-exhausted pipeline run currently realizes one final user message
    # that is never submitted to the evaluated assistant. Keep all feedback that
    # informed a later assistant response, but do not ask AITR to judge an
    # unanswered message for which the assistant had no interaction turn.
    last_assistant = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role == "assistant"
        ),
        None,
    )
    return tuple(messages[: last_assistant + 1]) if last_assistant is not None else tuple(messages)


def _resolve_assistant_model(events: tuple[dict[str, Any], ...], snapshot: dict[str, Any]) -> str:
    profile = snapshot.get("assistant_model_profile")
    snapshot_model = profile.get("model_id") if isinstance(profile, dict) else None
    event_models: set[str] = set()
    for event in events:
        if event.get("event_type") != "assistant_generation_completed":
            continue
        payload = event.get("payload")
        llm_call = payload.get("llm_call") if isinstance(payload, dict) else None
        model_id = llm_call.get("model_id") if isinstance(llm_call, dict) else None
        if isinstance(model_id, str) and model_id:
            event_models.add(model_id)
    if len(event_models) > 1:
        raise TraceLoadError(f"assistant model changed within one episode: {sorted(event_models)}")
    event_model = next(iter(event_models), None)
    if isinstance(snapshot_model, str) and snapshot_model:
        if event_model is not None and event_model != snapshot_model:
            raise TraceLoadError(
                f"assistant model mismatch: snapshot={snapshot_model!r}, events={event_model!r}"
            )
        return snapshot_model
    return event_model or "unknown"


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result
