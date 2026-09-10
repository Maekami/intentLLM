"""Small role prompts and reproducible snapshots using the existing pipeline hook."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from assistant.config import GP_ERROR_CODES
from assistant.exceptions import ConfigurationError

from .events import digest


@dataclass(frozen=True)
class RolePrompt:
    name: str
    system: str
    path: str
    modes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) - {"name", "system", "modes"}:
            raise ConfigurationError(f"Invalid GP v2 prompt fields: {path}")
        if any(not isinstance(raw.get(k), str) or not raw[k].strip() for k in ("name", "system")):
            raise ConfigurationError(f"GP v2 prompt requires nonempty name/system: {path}")
        modes = raw.get("modes", {})
        if not isinstance(modes, dict) or any(
            not isinstance(v, str) or not v.strip() for v in modes.values()
        ):
            raise ConfigurationError(f"Invalid GP prompt mode rules: {path}")
        return cls(name=raw["name"], system=raw["system"].strip(), path=str(path), modes=modes)

    def system_for(self, mode):
        if mode != "normal" and mode not in self.modes:
            raise ConfigurationError(f"Prompt {self.name} does not implement mode {mode}")
        return self.modes[mode].strip() if mode != "normal" else self.system

    @property
    def hash(self):
        return digest({"name": self.name, "system": self.system, "modes": self.modes})


def load_prompts(profile):
    settings = profile.goal_progression
    if settings is None:
        raise ConfigurationError("missing goal_progression settings")
    result = {role: RolePrompt.load(settings.prompts[role]) for role in settings.roles}
    if settings.variant == "no_tracker":
        for role in ("intra", "inter"):
            result[role].system_for("no_tracker")
    if settings.variant == "no_intra":
        result["generator"].system_for("no_intra")
    return result


def snapshot(profile):
    settings = profile.goal_progression
    if settings is None:
        return None
    return {
        **settings.model_dump(mode="json"),
        "roles": list(settings.roles),
        "generation": {
            role: profile.generation["assistant" if role == "generator" else role].model_dump(
                mode="json"
            )
            for role in settings.roles
        },
        "effective_retry_limits": {
            role: {code: settings.retry_limit(role, code) for code in sorted(GP_ERROR_CODES)}
            for role in (*settings.roles, "runtime")
        },
        "prompts": {
            role: {
                "name": prompt.name,
                "path": prompt.path,
                "hash": prompt.hash,
                "system": prompt.system,
                "modes": prompt.modes,
            }
            for role, prompt in load_prompts(profile).items()
        },
    }
