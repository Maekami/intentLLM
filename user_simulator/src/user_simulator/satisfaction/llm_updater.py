import json

from user_simulator.config import GenerationSettings
from user_simulator.domain.dag import DagNode
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import SatisfactionUpdateResult
from user_simulator.domain.state import EpisodeState
from user_simulator.exceptions import SatisfactionOutputError
from user_simulator.llm.base import StructuredLLMClient
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.satisfaction.base import SatisfactionUpdater


class LLMSatisfactionUpdater(SatisfactionUpdater):
    def __init__(
        self,
        client: StructuredLLMClient,
        generation: GenerationSettings,
        *,
        prompt_path: str = "configs/prompts/satisfaction.yaml",
        model_profile_name: str = "deepseek_v4_pro",
        semantic_attempts: int = 3,
    ) -> None:
        self.client = client
        self.generation = generation
        self.prompt = PromptTemplate.load(prompt_path)
        self.model_profile_name = model_profile_name
        self.semantic_attempts = semantic_attempts

    async def update(
        self,
        *,
        history: list[ChatMessage],
        latest_assistant_response: str,
        state: EpisodeState,
        exposed_nodes: list[DagNode],
    ) -> SatisfactionUpdateResult:
        context = json.dumps(
            {
                "conversation_history": [item.model_dump() for item in history],
                "latest_assistant_response": latest_assistant_response,
                "exposed_nodes": [item.model_dump() for item in exposed_nodes],
                "previous_satisfaction": {
                    key: value.value for key, value in state.satisfaction.items()
                },
            },
            ensure_ascii=False,
        )
        messages, metadata = self.prompt.render(context=context)
        metadata["model_profile"] = self.model_profile_name
        expected = [item.node_id for item in exposed_nodes]
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
                response_model=SatisfactionUpdateResult,
                schema_name="satisfaction_update_result",
                generation=self.generation,
                prompt_metadata={**metadata, "semantic_retry_count": attempt},
            )
            received = [item.node_id for item in result.updates]
            valid = (
                "END" not in received
                and len(received) == len(set(received))
                and len(received) == len(expected)
                and set(received) == set(expected)
            )
            if valid:
                return result
            last_problem = (
                f"updates must contain each exposed intent exactly once and no END; "
                f"expected {expected}, got {received}"
            )
        raise SatisfactionOutputError(
            f"satisfaction semantic validation failed after {self.semantic_attempts} "
            f"attempts: {last_problem}"
        )
