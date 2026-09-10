"""Finite producer-owned violations and per-turn independent retry quotas."""

from dataclasses import dataclass, field

from assistant.config import GP_ERROR_CODES, GP_ROLES, GoalProgressionSettings
from assistant.exceptions import InvalidModelResponseError


@dataclass
class ContractError(Exception):
    code: str
    owner: str
    detail: str
    field_path: str = ""
    affected_units: list[str] = field(default_factory=list)
    recoverable: bool = True
    artifact_id: str | None = None
    input_version: str | None = None

    def __post_init__(self):
        if self.code not in GP_ERROR_CODES or self.owner not in GP_ROLES:
            raise ValueError("unregistered GP error/owner")
        Exception.__init__(self, f"{self.owner}:{self.code}: {self.detail}")

    def payload(self):
        return {
            "code": self.code,
            "owner": self.owner,
            "detail": self.detail,
            "field_path": self.field_path,
            "affected_units": list(self.affected_units),
            "recoverable": self.recoverable,
            "artifact_id": self.artifact_id,
            "input_version": self.input_version,
        }


class RecoveryExhausted(InvalidModelResponseError):
    def __init__(self, error: ContractError):
        self.error = error
        super().__init__(f"GP recovery exhausted: {error}")


class RetryLedger:
    """One instance per transaction; artifact revisions never reset quotas."""

    def __init__(self, settings: GoalProgressionSettings):
        self.settings = settings
        self.counts: dict[tuple[str, str], int] = {}

    def consume(self, errors: list[ContractError]) -> dict[str, int]:
        unique = {(e.owner, e.code): e for e in errors}
        for key, error in unique.items():
            if not error.recoverable or self.counts.get(key, 0) >= self.settings.retry_limit(*key):
                raise RecoveryExhausted(error)
        for key in unique:
            self.counts[key] = self.counts.get(key, 0) + 1
        return self.snapshot()

    def snapshot(self) -> dict[str, int]:
        return {f"{role}:{code}": n for (role, code), n in sorted(self.counts.items())}
