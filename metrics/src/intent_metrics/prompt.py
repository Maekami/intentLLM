from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

from intent_metrics.config import PROJECT_ROOT
from intent_metrics.models import ConversationMessage


@dataclass(frozen=True)
class AITRPromptTemplate:
    name: str
    version: str
    system: str
    user_template: str
    path: str

    @classmethod
    def load(cls, path: str | Path) -> AITRPromptTemplate:
        prompt_path = Path(path).expanduser().resolve()
        raw = _read_yaml(prompt_path)
        required = {"name", "version", "system", "user_template"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"{prompt_path} is missing prompt fields: {sorted(missing)}")
        unknown = set(raw) - required
        if unknown:
            raise ValueError(f"{prompt_path} contains unknown prompt fields: {sorted(unknown)}")
        prompt = cls(
            name=_nonempty_text(raw["name"], "name", prompt_path),
            version=_nonempty_text(raw["version"], "version", prompt_path),
            system=_nonempty_text(raw["system"], "system", prompt_path),
            user_template=_nonempty_text(raw["user_template"], "user_template", prompt_path),
            path=str(prompt_path),
        )
        prompt.validate_contract()
        return prompt

    @property
    def hash(self) -> str:
        canonical = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "system": self.system,
                "user_template": self.user_template,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate_contract(self) -> None:
        placeholders = {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.user_template)
            if field_name
        }
        if placeholders != {"conversation"}:
            raise ValueError(
                f"{self.path} user_template placeholders must be exactly "
                f"['conversation']; found {sorted(placeholders)}"
            )

    def build_messages(self, messages: tuple[ConversationMessage, ...]) -> list[dict[str, str]]:
        conversation = render_conversation(messages)
        return [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": self.user_template.format(conversation=conversation),
            },
        ]


def load_aitr_prompt(path: str | Path | None = None) -> AITRPromptTemplate:
    if path is None:
        path = PROJECT_ROOT / "configs/prompts/aitr.yaml"
    return AITRPromptTemplate.load(path)


def render_conversation(messages: tuple[ConversationMessage, ...]) -> str:
    """Render only visible user/assistant content for the AITR judge."""

    blocks: list[str] = []
    for message in messages:
        if message.role not in {"user", "assistant"}:
            continue
        label = "User" if message.role == "user" else "Assistant"
        blocks.append(f"{label}: {escape(message.content, quote=False)}")
    return "\n\n".join(blocks)


def build_aitr_messages(
    messages: tuple[ConversationMessage, ...],
    *,
    prompt: AITRPromptTemplate | None = None,
) -> list[dict[str, str]]:
    return (prompt or load_aitr_prompt()).build_messages(messages)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"cannot read AITR prompt {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse AITR prompt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def _nonempty_text(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} prompt field {field!r} must be non-empty text")
    return value


# Backward-compatible public version, derived from YAML rather than hard-coded.
AITR_PROMPT_VERSION = load_aitr_prompt().version

__all__ = [
    "AITR_PROMPT_VERSION",
    "AITRPromptTemplate",
    "build_aitr_messages",
    "load_aitr_prompt",
    "render_conversation",
]
