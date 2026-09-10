"""Recount delivered replies for every baseline with one explicitly pinned tokenizer.

No model calls, hidden-state reads, or edits to historical run artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from intent_metrics.errors import MetricDataError
from intent_metrics.traces import discover_run_dirs


def recount_runs(run_dirs: list[Path], tokenizer: Any) -> dict[str, Any]:
    episodes = []
    for directory in run_dirs:
        path = directory / "transcript.jsonl"
        if not path.is_file():
            raise MetricDataError(f"missing visible transcript: {path}")
        tokens = 0
        turns = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            message = json.loads(line)
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                raise MetricDataError(f"invalid assistant content: {path}")
            tokens += len(tokenizer.encode(content, add_special_tokens=False).ids)
            turns += 1
        episodes.append(
            {"run_dir": str(directory), "visible_tokens": tokens, "delivered_turns": turns}
        )
    return {
        "episode_count": len(episodes),
        "avg_visible_tokens": sum(e["visible_tokens"] for e in episodes) / len(episodes)
        if episodes
        else None,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dirs", nargs="+", type=Path, help="One or more run/batch directories"
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Existing local tokenizer.json; never downloaded",
    )
    parser.add_argument("--output", required=True, type=Path, help="Independent JSON report")
    args = parser.parse_args()
    from tokenizers import Tokenizer

    path = args.tokenizer.expanduser().resolve()
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    report = {
        "definition": "mean over episodes of sum of tokens in delivered assistant text; no per-turn division",
        "tokenizer_path": str(path),
        "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "add_special_tokens": False,
        "groups": {
            str(source.resolve()): recount_runs(discover_run_dirs(source), tokenizer)
            for source in args.input_dirs
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
