import json

from user_simulator.config import GenerationSettings
from user_simulator.controller.base import Controller
from user_simulator.domain.dag import DagNode
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import ControllerResult
from user_simulator.domain.state import EpisodeState
from user_simulator.exceptions import ControllerOutputError, StructuredOutputError
from user_simulator.llm.base import StructuredLLMClient
from user_simulator.llm.prompt import PromptTemplate


class LLMController(Controller):
    def __init__(
        self,
        client: StructuredLLMClient,
        generation: GenerationSettings,
        *,
        prompt: PromptTemplate,
        model_profile_name: str = "deepseek_v4_flash_0731",
        semantic_attempts: int = 3,
    ) -> None:
        self.client = client
        self.generation = generation
        self.prompt = prompt
        self.model_profile_name = model_profile_name
        self.semantic_attempts = semantic_attempts
        self.semantic_events: list[dict] = []

    async def decide(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        candidates: list[DagNode],
    ) -> ControllerResult:
        context = controller_context(
            history=history,
            latest_assistant_response=latest_assistant_response,
            state=state,
            candidates=candidates,
        )
        expected = [item.node_id for item in candidates]
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
                candidate_order=json.dumps(expected),
                correction=corrective_message,
            )
            metadata["model_profile"] = self.model_profile_name
            try:
                result = await self.client.generate_structured(
                    messages=call_messages,
                    response_model=ControllerResult,
                    schema_name=self.prompt.schema.name,
                    generation=self.generation,
                    prompt_metadata={**metadata, "semantic_retry_count": attempt},
                )
            except StructuredOutputError as exc:
                raise ControllerOutputError(str(exc)) from exc
            received = [item.node_id for item in result.decisions]
            if received == expected and len(received) == len(set(received)):
                self.semantic_events.append(
                    {
                        "semantic_attempt": attempt + 1,
                        "semantic_error": None,
                        "corrective_message": corrective_message,
                    }
                )
                return result
            last_problem = f"Candidate order mismatch. Required: {expected}. Returned: {received}"
            self.semantic_events.append(
                {
                    "semantic_attempt": attempt + 1,
                    "semantic_error": last_problem,
                    "corrective_message": corrective_message,
                }
            )
        raise ControllerOutputError(
            f"controller semantic validation failed after {self.semantic_attempts} "
            f"attempts: {last_problem}"
        )


def controller_context(
    *,
    history: list[ChatMessage],
    latest_assistant_response: str,
    state: EpisodeState,
    candidates: list[DagNode],
) -> str:
    return _labeled_context(
        [
            (
                "VISIBLE CONVERSATION",
                [item.model_dump() for item in history],
            ),
            ("LATEST ASSISTANT MESSAGE", latest_assistant_response),
            ("CURRENT FRONTIER", state.current_frontier),
            ("EXPOSED NODE IDS", state.exposed_nodes),
            (
                "CURRENT SATISFACTION STATES",
                {key: value.value for key, value in state.satisfaction.items()},
            ),
            (
                "ORDERED CANDIDATE NODE DETAILS",
                [item.model_dump() for item in candidates],
            ),
        ]
    )


def _labeled_context(sections: list[tuple[str, object]]) -> str:
    return "\n\n".join(
        f"{label}\n{json.dumps(value, ensure_ascii=False, indent=2)}" for label, value in sections
    )
