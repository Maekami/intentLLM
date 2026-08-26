import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SystemPrompt:
    name: str
    system: str
    path: str

    @classmethod
    def load(cls, path: str | Path) -> "SystemPrompt":
        prompt_path = Path(path)
        with prompt_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path} must contain a YAML mapping")
        required = {"name", "system"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"{path} is missing prompt fields: {sorted(missing)}")
        unknown = set(raw) - required
        if unknown:
            raise ValueError(f"{path} contains unknown prompt fields: {sorted(unknown)}")
        system = str(raw["system"]).strip()
        if not system:
            raise ValueError(f"{path} contains an empty system prompt")
        return cls(name=str(raw["name"]), system=system, path=str(prompt_path))

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.system.encode()).hexdigest()
