import json

from user_simulator.config import GenerationSettings
from user_simulator.controller.base import Controller
from user_simulator.domain.dag import DagNode
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import ControllerResult
from user_simulator.domain.state import EpisodeState
from user_simulator.exceptions import ControllerOutputError
from user_simulator.llm.base import StructuredLLMClient
from user_simulator.llm.prompt import PromptTemplate


class LLMController(Controller):
    def __init__(
        self,
        client: StructuredLLMClient,
        generation: GenerationSettings,
        *,
        prompt_path: str = "configs/prompts/controller.yaml",
        model_profile_name: str = "deepseek_v4_pro",
        semantic_attempts: int = 3,
    ) -> None:
        self.client = client
        self.generation = generation
        self.prompt = PromptTemplate.load(prompt_path)
        self.model_profile_name = model_profile_name
        self.semantic_attempts = semantic_attempts

    async def decide(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        candidates: list[DagNode],
        has_end_edge: bool,
    ) -> ControllerResult:
        context = json.dumps(
            {
                "conversation_history": [item.model_dump() for item in history],
                "latest_assistant_response": latest_assistant_response,
                "current_frontier": state.current_frontier,
                "exposed_nodes": state.exposed_nodes,
                "satisfaction": {key: value.value for key, value in state.satisfaction.items()},
                "sorted_outgoing_candidates": [item.model_dump() for item in candidates],
                "has_outgoing_end_edge": has_end_edge,
            },
            ensure_ascii=False,
        )
        messages, metadata = self.prompt.render(context=context)
        metadata["model_profile"] = self.model_profile_name
        expected = [item.node_id for item in candidates]
        last_problem = ""
        for attempt in range(self.semantic_attempts):
            call_messages = list(messages)
            if last_problem:
                call_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Correction required: "
                            + last_problem
                            + ". Return a complete corrected result."
                        ),
                    }
                )
            result = await self.client.generate_structured(
                messages=call_messages,
                response_model=ControllerResult,
                schema_name="controller_result",
                generation=self.generation,
                prompt_metadata={**metadata, "semantic_retry_count": attempt},
            )
            received = [item.node_id for item in result.decisions]
            if len(received) == len(set(received)) and set(received) == set(expected):
                return result
            last_problem = (
                f"decisions must contain each candidate exactly once; expected "
                f"{expected}, got {received}"
            )
        raise ControllerOutputError(
            f"controller semantic validation failed after {self.semantic_attempts} "
            f"attempts: {last_problem}"
        )
