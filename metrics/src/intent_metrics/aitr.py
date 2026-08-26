from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from user_simulator.config import EnvironmentSettings
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient

from intent_metrics.config import (
    AITRConfig,
    AITREnvironmentSettings,
    load_aitr_config,
)
from intent_metrics.errors import AITREvaluationError
from intent_metrics.models import ConversationMessage
from intent_metrics.prompt import AITR_PROMPT_VERSION, load_aitr_prompt


class AITRJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reason: str = Field(min_length=1, max_length=4000)
    score: float = Field(ge=1.0, le=3.0)


@dataclass(frozen=True)
class AITRJudgeResponse:
    judgment: AITRJudgment
    raw_response: dict[str, Any]
    metadata: dict[str, Any]


class AITRJudge(Protocol):
    model_id: str
    reasoning_effort: str
    config_hash: str
    prompt_version: str
    prompt_hash: str

    async def judge(self, messages: tuple[ConversationMessage, ...]) -> AITRJudgeResponse: ...


class OpenRouterAITRJudge:
    """Strict structured-output AITR judge using the simulator's proven client."""

    def __init__(
        self,
        config: AITRConfig | None = None,
        environment: AITREnvironmentSettings | None = None,
        *,
        client: OpenRouterStructuredClient | None = None,
    ) -> None:
        self.config = config or load_aitr_config()
        self.environment = environment or AITREnvironmentSettings()
        self.model_id = self.config.model_id
        self.reasoning_effort = self.config.reasoning.effort
        self.config_hash = self.config.hash
        self.prompt = load_aitr_prompt(self.config.prompt)
        self.prompt_version = self.prompt.version
        self.prompt_hash = self.prompt.hash
        simulator_environment = EnvironmentSettings(
            openrouter_api_key=self.environment.openrouter_api_key,
            openrouter_http_referer=self.environment.openrouter_http_referer,
            openrouter_app_title=self.environment.openrouter_app_title,
        )
        self.client = client or OpenRouterStructuredClient(self.config, simulator_environment)

    async def judge(self, messages: tuple[ConversationMessage, ...]) -> AITRJudgeResponse:
        if not messages:
            raise AITREvaluationError("AITR requires a non-empty natural conversation")
        try:
            judgment = await self.client.generate_structured(
                messages=self.prompt.build_messages(messages),
                response_model=AITRJudgment,
                schema_name="aitr_judgment",
                generation=self.config.generation["aitr"],
                prompt_metadata={
                    "prompt_name": self.prompt.name,
                    "prompt_path": self.prompt.path,
                    "prompt_version": self.prompt.version,
                    "prompt_hash": self.prompt.hash,
                },
            )
        except Exception as exc:
            raise AITREvaluationError(
                f"AITR judgment failed after configured retries: {type(exc).__name__}: {exc}"
            ) from exc
        metadata = {
            key: value
            for key, value in self.client.last_call_metadata.items()
            if key not in {"messages", "raw_response", "invalid_raw_response"}
        }
        raw = judgment.model_dump(mode="json")
        return AITRJudgeResponse(judgment=judgment, raw_response=raw, metadata=metadata)


def score_to_aitr(score: float) -> tuple[float, float]:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("AITR raw score must be numeric")
    value = float(score)
    if not math.isfinite(value) or not 1.0 <= value <= 3.0:
        raise ValueError("AITR raw score must be finite and within [1.0, 3.0]")
    normalized = (value - 1.0) / 2.0
    return normalized, normalized * 100.0


__all__ = [
    "AITR_PROMPT_VERSION",
    "AITRConfig",
    "AITREnvironmentSettings",
    "AITRJudge",
    "AITRJudgeResponse",
    "AITRJudgment",
    "OpenRouterAITRJudge",
    "load_aitr_config",
    "score_to_aitr",
]
