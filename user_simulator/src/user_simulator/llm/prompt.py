import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user_template: str

    @classmethod
    def load(cls, path: str | Path) -> "PromptTemplate":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls(
            name=str(raw["name"]),
            version=int(raw["version"]),
            system=str(raw["system"]),
            user_template=str(raw["user_template"]),
        )

    def render(self, **values: str) -> tuple[list[dict[str, str]], dict[str, str | int]]:
        user = self.user_template.format(**values)
        rendered = f"{self.system}\n{user}"
        metadata: dict[str, str | int] = {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "rendered_prompt_hash": hashlib.sha256(rendered.encode()).hexdigest(),
        }
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ], metadata
