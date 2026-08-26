from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intent_metrics.errors import CacheError
from intent_metrics.models import ConversationMessage

_CACHE_FORMAT_VERSION = 1


class AITRCacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    conversation_hash: str = Field(min_length=64, max_length=64)
    judge_model: str = Field(min_length=1)
    reasoning_effort: str = Field(min_length=1)
    judge_config_hash: str | None = Field(default=None, min_length=64, max_length=64)
    prompt_version: str = Field(min_length=1)
    prompt_hash: str | None = Field(default=None, min_length=64, max_length=64)
    raw_judge_response: dict[str, Any]
    raw_score: float = Field(ge=1.0, le=3.0)
    normalized_score: float = Field(ge=0.0, le=1.0)
    aitr: float = Field(ge=0.0, le=100.0)
    reason: str = Field(min_length=1)
    judge_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class AITRCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._entries = self._load()

    def get(
        self,
        *,
        episode_id: str,
        conversation_hash: str,
        judge_model: str,
        reasoning_effort: str,
        judge_config_hash: str,
        prompt_version: str,
        prompt_hash: str,
    ) -> AITRCacheRecord | None:
        key = cache_key(
            episode_id=episode_id,
            conversation_hash=conversation_hash,
            judge_model=judge_model,
            reasoning_effort=reasoning_effort,
            judge_config_hash=judge_config_hash,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
        )
        record = self._entries.get(key)
        if record is None:
            return None
        expected = (
            episode_id,
            conversation_hash,
            judge_model,
            reasoning_effort,
            judge_config_hash,
            prompt_version,
            prompt_hash,
        )
        actual = (
            record.episode_id,
            record.conversation_hash,
            record.judge_model,
            record.reasoning_effort,
            record.judge_config_hash,
            record.prompt_version,
            record.prompt_hash,
        )
        if actual != expected:
            raise CacheError(f"AITR cache key collision or corrupt entry in {self.path}")
        return record

    def put(self, record: AITRCacheRecord) -> None:
        key = cache_key(
            episode_id=record.episode_id,
            conversation_hash=record.conversation_hash,
            judge_model=record.judge_model,
            reasoning_effort=record.reasoning_effort,
            judge_config_hash=record.judge_config_hash or ("0" * 64),
            prompt_version=record.prompt_version,
            prompt_hash=record.prompt_hash or ("0" * 64),
        )
        self._entries[key] = record
        self._save()

    def _load(self) -> dict[str, AITRCacheRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("format_version") != _CACHE_FORMAT_VERSION:
                raise CacheError(f"unsupported AITR cache format in {self.path}")
            entries = raw.get("entries")
            if not isinstance(entries, dict):
                raise CacheError(f"AITR cache entries must be a mapping in {self.path}")
            return {
                str(key): AITRCacheRecord.model_validate(value) for key, value in entries.items()
            }
        except CacheError:
            raise
        except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise CacheError(f"cannot load AITR cache {self.path}: {exc}") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _CACHE_FORMAT_VERSION,
            "entries": {
                key: record.model_dump(mode="json") for key, record in sorted(self._entries.items())
            },
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError as exc:
            raise CacheError(f"cannot save AITR cache {self.path}: {exc}") from exc


def conversation_hash(messages: tuple[ConversationMessage, ...]) -> str:
    canonical = json.dumps(
        [{"role": item.role, "content": item.content} for item in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_key(
    *,
    episode_id: str,
    conversation_hash: str,
    judge_model: str,
    reasoning_effort: str,
    judge_config_hash: str,
    prompt_version: str,
    prompt_hash: str,
) -> str:
    canonical = json.dumps(
        {
            "episode_id": episode_id,
            "conversation_hash": conversation_hash,
            "judge_model": judge_model,
            "reasoning_effort": reasoning_effort,
            "judge_config_hash": judge_config_hash,
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
