import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import ClassVar

import yaml

from user_simulator.llm.schema_registry import SchemaSpec, resolve_schema


@dataclass(frozen=True)
class PromptTemplate:
    _ALLOWED_PLACEHOLDERS: ClassVar[dict[str, set[str]]] = {
        "controller_result": {"context", "candidate_order", "correction"},
        "satisfaction_update_result": {
            "context",
            "exposed_node_order",
            "correction",
        },
        "user_generation_result": {
            "context",
            "selected_node_order",
            "correction",
        },
    }

    name: str
    schema_name: str
    system: str
    user_template: str
    path: str

    @classmethod
    def load(cls, path: str | Path) -> "PromptTemplate":
        prompt_path = Path(path)
        with prompt_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        required = {"name", "schema_name", "system", "user_template"}
        missing = required - set(raw or {})
        if missing:
            raise ValueError(f"{path} is missing prompt fields: {sorted(missing)}")
        unknown = set(raw or {}) - required
        if unknown:
            raise ValueError(f"{path} contains unknown prompt fields: {sorted(unknown)}")
        prompt = cls(
            name=str(raw["name"]),
            schema_name=str(raw["schema_name"]),
            system=str(raw["system"]),
            user_template=str(raw["user_template"]),
            path=str(prompt_path),
        )
        prompt.validate_contract()
        return prompt

    @property
    def schema(self) -> SchemaSpec:
        return resolve_schema(self.schema_name)

    @property
    def placeholders(self) -> set[str]:
        return {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.user_template)
            if field_name
        }

    def validate_contract(self) -> None:
        allowed = self._ALLOWED_PLACEHOLDERS.get(self.schema.name)
        if allowed is None:
            raise ValueError(f"no placeholder contract for schema {self.schema.name}")
        unknown = self.placeholders - allowed
        if unknown:
            raise ValueError(f"{self.path} contains unknown placeholders: {sorted(unknown)}")
        missing = allowed - self.placeholders
        if missing:
            raise ValueError(f"{self.path} is missing required placeholders: {sorted(missing)}")

    def render(self, **values: str) -> tuple[list[dict[str, str]], dict[str, str]]:
        missing = self.placeholders - set(values)
        if missing:
            raise ValueError(f"missing render values for {self.name}: {sorted(missing)}")
        user = self.user_template.format(**values)
        rendered = f"{self.system}\n{user}"
        metadata = {
            "prompt_name": self.name,
            "prompt_path": self.path,
            "prompt_hash": hashlib.sha256(rendered.encode()).hexdigest(),
            "schema_name": self.schema_name,
            "schema_hash": self.schema.hash,
        }
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ], metadata
