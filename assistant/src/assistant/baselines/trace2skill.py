from collections.abc import Sequence
from typing import Any

from assistant.baselines.base import AssistantBaseline
from assistant.domain.messages import ChatMessage
from assistant.skill import StaticSkill


class Trace2SkillBaseline(AssistantBaseline):
    """Static test-time baseline produced by the offline Trace2Skill pipeline."""

    name = "trace2skill"

    def __init__(self, skill: StaticSkill) -> None:
        self.skill = skill

    def build_messages(self, history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.skill.system},
            *[message.model_dump(mode="json", exclude_none=True) for message in history],
        ]
