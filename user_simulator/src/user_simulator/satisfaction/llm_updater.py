import json
from typing import cast

from user_simulator.config import GenerationSettings
from user_simulator.domain.dag import DagNode
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import SatisfactionUpdateResult
from user_simulator.domain.state import EpisodeState
from user_simulator.exceptions import SatisfactionOutputError, StructuredOutputError
from user_simulator.llm.base import StructuredLLMClient
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.llm.schemas import SatisfactionUpdateResultV2
from user_simulator.satisfaction.base import SatisfactionUpdater


class LLMSatisfactionUpdater(SatisfactionUpdater):
    def __init__(
        self,
        client: StructuredLLMClient,
        generation: GenerationSettings,
        *,
        prompt: PromptTemplate | None = None,
        prompt_path: str = "configs/prompts/satisfaction_v2.yaml",
        model_profile_name: str = "deepseek_v4_pro",
        semantic_attempts: int = 3,
    ) -> None:
        self.client = client
        self.generation = generation
        self.prompt = prompt or PromptTemplate.load(prompt_path)
        self.model_profile_name = model_profile_name
        self.semantic_attempts = semantic_attempts
        self.semantic_events: list[dict] = []

    async def update(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        exposed_nodes: list[DagNode],
    ) -> SatisfactionUpdateResult:
        context = _labeled_context(
            [
                (
                    "VISIBLE CONVERSATION WITH ROLES",
                    [item.model_dump() for item in history],
                ),
                ("LATEST ASSISTANT RESPONSE", latest_assistant_response),
                (
                    "EXPOSED NODE DETAILS IN REQUIRED ORDER",
                    [item.model_dump() for item in exposed_nodes],
                ),
                (
                    "PREVIOUS SATISFACTION STATES",
                    {key: value.value for key, value in state.satisfaction.items()},
                ),
            ]
        )
        expected = [item.node_id for item in exposed_nodes]
        self.semantic_events = []
        last_problem = ""
        for attempt in range(self.semantic_attempts):
            corrective_message = ""
            if last_problem:
                corrective_message = (
                    f"Correction required: {last_problem}. Return a complete corrected result."
                )
            call_messages, metadata = self.prompt.render(
                context=context,
                exposed_node_order=json.dumps(expected),
                correction=corrective_message,
            )
            metadata["model_profile"] = self.model_profile_name
            try:
                result = cast(
                    SatisfactionUpdateResultV2,
                    await self.client.generate_structured(
                        messages=call_messages,
                        response_model=self.prompt.schema.model,
                        schema_name=self.prompt.schema.name,
                        schema_version=self.prompt.schema.version,
                        generation=self.generation,
                        prompt_metadata={**metadata, "semantic_retry_count": attempt},
                    ),
                )
            except StructuredOutputError as exc:
                raise SatisfactionOutputError(str(exc)) from exc
            received = [item.node_id for item in result.updates]
            valid = (
                "END" not in received
                and len(received) == len(set(received))
                and received == expected
                and all(_specific_reason(item.reason) for item in result.updates)
            )
            if valid:
                self.semantic_events.append(
                    {
                        "semantic_attempt": attempt + 1,
                        "semantic_error": None,
                        "corrective_message": corrective_message,
                    }
                )
                return result
            if received != expected:
                last_problem = (
                    f"Exposed-node order mismatch. Required: {expected}. Returned: {received}"
                )
            else:
                generic = [
                    item.node_id for item in result.updates if not _specific_reason(item.reason)
                ]
                last_problem = (
                    "Reasons must identify assistant-provided evidence or a concrete "
                    f"missing requirement; generic reasons returned for {generic}"
                )
            self.semantic_events.append(
                {
                    "semantic_attempt": attempt + 1,
                    "semantic_error": last_problem,
                    "corrective_message": corrective_message,
                }
            )
        raise SatisfactionOutputError(
            f"satisfaction semantic validation failed after {self.semantic_attempts} "
            f"attempts: {last_problem}"
        )


def _specific_reason(reason: str) -> bool:
    words = [word.strip(".,:;!?").lower() for word in reason.split()]
    generic = {
        "done",
        "satisfied",
        "unsatisfied",
        "partial",
        "partially",
        "not done",
        "no",
        "yes",
    }
    return len(words) >= 4 and reason.strip().lower() not in generic


def _labeled_context(sections: list[tuple[str, object]]) -> str:
    return "\n\n".join(
        f"{label}\n{json.dumps(value, ensure_ascii=False, indent=2)}" for label, value in sections
    )
