from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from assistant.config import MemorySettings
from assistant.memory.models import MemoryEntry, MemoryUpdateResult

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux is used for benchmark runs
    fcntl = None  # type: ignore[assignment]


class MemoryStoreError(RuntimeError):
    """Raised when a persisted memory file cannot be read or updated safely."""


class JsonMemoryStore:
    """Atomic, process-safe JSON store for an evolving task stream."""

    format_version = 1

    def __init__(self, settings: MemorySettings) -> None:
        self.settings = settings
        self.path = settings.path
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")

    def entries(self) -> list[MemoryEntry]:
        with self._locked():
            return self._read_unlocked()

    def upsert(self, entry: MemoryEntry) -> MemoryUpdateResult:
        return self.upsert_many([entry])[0]

    def upsert_many(
        self,
        new_entries: Iterable[MemoryEntry],
        *,
        expected_entries: Iterable[MemoryEntry] | None = None,
    ) -> list[MemoryUpdateResult]:
        """Apply ordered updates with one read-lock-write transaction."""

        pending = list(new_entries)
        expected = list(expected_entries) if expected_entries is not None else None
        if not pending and expected is None:
            return []
        with self._locked():
            entries = self._read_unlocked()
            if expected is not None and [item.to_dict() for item in entries] != [
                item.to_dict() for item in expected
            ]:
                raise MemoryStoreError(
                    f"memory store {self.path} changed while an evolution mini-batch was running"
                )
            if not pending:
                return []
            results: list[MemoryUpdateResult] = []
            changed = False
            for entry in pending:
                existing_index = next(
                    (index for index, item in enumerate(entries) if item.task_id == entry.task_id),
                    None,
                )
                if existing_index is not None:
                    entries[existing_index] = entry
                    reason = "updated_existing_task"
                elif len(entries) >= self.settings.max_entries:
                    if not self.settings.prune_oldest_when_full:
                        results.append(
                            MemoryUpdateResult(
                                stored=False,
                                task_id=entry.task_id,
                                path=str(self.path),
                                entry_count=len(entries),
                                reason="capacity_reached",
                            )
                        )
                        continue
                    remove_count = len(entries) - self.settings.max_entries + 1
                    entries = entries[remove_count:]
                    entries.append(entry)
                    reason = "added_after_pruning_oldest"
                else:
                    entries.append(entry)
                    reason = "added"
                changed = True
                results.append(
                    MemoryUpdateResult(
                        stored=True,
                        task_id=entry.task_id,
                        path=str(self.path),
                        entry_count=len(entries),
                        reason=reason,
                    )
                )
            if changed:
                self._write_unlocked(entries)
            return results

    def skipped(self, *, task_id: str | None, reason: str) -> MemoryUpdateResult:
        return MemoryUpdateResult(
            stored=False,
            task_id=task_id,
            path=str(self.path),
            entry_count=len(self.entries()),
            reason=reason,
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except MemoryStoreError:
            raise
        except OSError as exc:
            raise MemoryStoreError(f"cannot access memory store {self.path}: {exc}") from exc

    def _read_unlocked(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryStoreError(f"cannot read memory store {self.path}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
            raise MemoryStoreError(
                f"memory store {self.path} must contain an object with an entries list"
            )
        format_version = value.get("format_version")
        if format_version not in (None, self.format_version):
            raise MemoryStoreError(
                f"memory store {self.path} uses unsupported format_version={format_version!r}"
            )
        stored_framework = value.get("framework")
        if stored_framework not in (None, self.settings.framework):
            raise MemoryStoreError(
                f"memory store {self.path} belongs to framework={stored_framework!r}, "
                f"not {self.settings.framework!r}"
            )
        try:
            return [MemoryEntry.from_dict(item) for item in value["entries"]]
        except (TypeError, ValueError) as exc:
            raise MemoryStoreError(f"invalid entry in memory store {self.path}: {exc}") from exc

    def _write_unlocked(self, entries: list[MemoryEntry]) -> None:
        payload: dict[str, Any] = {
            "format_version": self.format_version,
            "framework": self.settings.framework,
            "entries": [entry.to_dict() for entry in entries],
            "stats": {
                "entry_count": len(entries),
                "successful_entries": sum(entry.is_successful for entry in entries),
            },
        }
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise MemoryStoreError(f"cannot write memory store {self.path}: {exc}") from exc
