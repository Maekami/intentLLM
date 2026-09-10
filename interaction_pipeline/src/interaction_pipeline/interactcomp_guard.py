from __future__ import annotations

from typing import Any

from assistant.config import GenerationSettings as AssistantGenerationSettings
from assistant.llm.base import GeneratedResponse
from pydantic import BaseModel, ConfigDict
from user_simulator.config import (
    EnvironmentSettings as SimulatorEnvironmentSettings,
)
from user_simulator.config import GenerationSettings as SimulatorGenerationSettings
from user_simulator.config import ModelProfile as SimulatorModelProfile
from user_simulator.llm.openrouter_client import OpenRouterStructuredClient


class _GuardDecisionPayload(BaseModel):
    """Wire contract equivalent to the official ``{ok, reason}`` JSON."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ok: bool
    reason: str = ""


class SimulatorProfileGuardClient:
    """Expose one simulator-profile structured client as an assistant chat client.

    The model/provider/routing/reasoning/retry/structured-output behavior all
    come from the YAML selected by ``--simulator-model-profile``. Only sampling
    parameters come from the shared InteractComp guard configuration, mirroring
    the reference runner's post-load temperature override.
    """

    def __init__(
        self,
        profile: SimulatorModelProfile,
        environment: SimulatorEnvironmentSettings,
        *,
        client: OpenRouterStructuredClient | None = None,
    ) -> None:
        self.profile = profile
        self._client = client or OpenRouterStructuredClient(profile, environment)
        self.last_call_metadata: dict[str, Any] = {}

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        generation: AssistantGenerationSettings,
    ) -> GeneratedResponse:
        normalized = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in messages
        ]
        simulator_generation = SimulatorGenerationSettings(
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            min_p=generation.min_p,
            presence_penalty=generation.presence_penalty,
            repetition_penalty=generation.repetition_penalty,
            max_completion_tokens=generation.max_completion_tokens,
        )
        try:
            result = await self._client.generate_structured(
                messages=normalized,
                response_model=_GuardDecisionPayload,
                schema_name="interactcomp_action_guard_decision",
                generation=simulator_generation,
                prompt_metadata={
                    "component": "interactcomp_action_guard",
                    "model_profile": self.profile.profile_name,
                },
            )
        finally:
            # The simulator client records structured/transport failure details
            # before raising. Mirror them so the assistant failure audit does
            # not lose the validator retry outcome.
            metadata = getattr(self._client, "last_call_metadata", {})
            self.last_call_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        content = result.model_dump_json()
        return GeneratedResponse(
            content=content,
            metadata=dict(self.last_call_metadata),
        )


def action_guard_model_metadata(
    *,
    profile: SimulatorModelProfile,
    source: str,
    source_path: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_path": source_path,
        "profile_name": profile.profile_name,
        "model_id": profile.model_id,
        "provider": profile.provider,
    }
