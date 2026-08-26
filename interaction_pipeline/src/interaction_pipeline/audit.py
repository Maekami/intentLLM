import json
from pathlib import Path


def render_human_audit(run_dir: str | Path) -> None:
    directory = Path(run_dir)
    _render_events(directory / "events.jsonl", directory / "events.txt")
    _render_transcript(directory / "transcript.jsonl", directory / "transcript.txt")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                value = {"parse_error": f"line {line_number}: {exc}", "raw": line.rstrip()}
            records.append(value)
    return records


def _render_events(source: Path, destination: Path) -> None:
    blocks: list[str] = []
    for index, record in enumerate(_read_jsonl(source), 1):
        header = (
            f"EVENT {index:04d} | {record.get('event_type', 'unknown')} | "
            f"turn {record.get('turn_index', '?')} | {record.get('timestamp', '')}"
        )
        payload = json.dumps(record.get("payload", {}), ensure_ascii=False, indent=2)
        blocks.append(f"{header}\n{'=' * len(header)}\n{payload}")
    destination.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def _render_transcript(source: Path, destination: Path) -> None:
    blocks: list[str] = []
    for record in _read_jsonl(source):
        role = str(record.get("role", "unknown")).upper()
        header = f"TURN {record.get('turn_index', '?')} | {role}"
        blocks.append(f"{header}\n{'-' * len(header)}\n{record.get('content', '')}")
    destination.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
