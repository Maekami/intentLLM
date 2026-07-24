import json

from user_simulator.config import GenerationSettings
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.domain.enums import RealizationMode, SatisfactionLevel
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import UserGenerationResult
from user_simulator.exceptions import UserGenerationError
from user_simulator.llm.base import StructuredLLMClient
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.realizer.base import UserRealizer


class LLMUserRealizer(UserRealizer):
    def __init__(
        self,
        client: StructuredLLMClient,
        clear_generation: GenerationSettings,
        abstract_generation: GenerationSettings,
        *,
        clear_prompt_path: str = "configs/prompts/user_clear.yaml",
        abstract_prompt_path: str = "configs/prompts/user_abstract.yaml",
        model_profile_name: str = "deepseek_v4_pro",
        abstract_client: StructuredLLMClient | None = None,
        abstract_model_profile_name: str | None = None,
    ) -> None:
        self.client = client
        self.clients = {
            RealizationMode.CLEAR: client,
            RealizationMode.ABSTRACT: abstract_client or client,
        }
        self.generations = {
            RealizationMode.CLEAR: clear_generation,
            RealizationMode.ABSTRACT: abstract_generation,
        }
        self.prompts = {
            RealizationMode.CLEAR: PromptTemplate.load(clear_prompt_path),
            RealizationMode.ABSTRACT: PromptTemplate.load(abstract_prompt_path),
        }
        self.model_profile_names = {
            RealizationMode.CLEAR: model_profile_name,
            RealizationMode.ABSTRACT: abstract_model_profile_name or model_profile_name,
        }
        self.last_call_metadata: dict = {}

    async def generate(
        self,
        *,
        selected_nodes: list[DagNode],
        unselected_unresolved_nodes: list[DagNode],
        satisfaction: dict[str, SatisfactionLevel],
        history: list[ChatMessage],
        latest_assistant_response: str | None,
        mode: RealizationMode,
        sample: Sample,
    ) -> UserGenerationResult:
        selected_ids = [item.node_id for item in selected_nodes]
        context = json.dumps(
            {
                "selected_nodes": [item.model_dump() for item in selected_nodes],
                "unselected_unresolved_nodes": [
                    item.model_dump() for item in unselected_unresolved_nodes
                ],
                "satisfaction": {key: value.value for key, value in satisfaction.items()},
                "conversation_history": [item.model_dump() for item in history],
                "latest_assistant_response": latest_assistant_response,
                "realization_mode": mode.value,
                "task_summary": sample.task_summary,
                "task_expectation": sample.task_expectation,
            },
            ensure_ascii=False,
        )
        correction = ""
        for attempt in range(2):
            messages, metadata = self.prompts[mode].render(context=context, correction=correction)
            metadata["model_profile"] = self.model_profile_names[mode]
            selected_client = self.clients[mode]
            result = await selected_client.generate_structured(
                messages=messages,
                response_model=UserGenerationResult,
                schema_name="user_generation_result",
                generation=self.generations[mode],
                prompt_metadata=metadata,
            )
            call_metadata = getattr(selected_client, "last_call_metadata", {})
            self.last_call_metadata = dict(call_metadata) if isinstance(call_metadata, dict) else {}
            errors = _generation_errors(result, selected_ids, mode)
            if not errors:
                return result
            correction = (
                "Correction required after invalid self-check: "
                + "; ".join(errors)
                + ". Regenerate the whole result."
            )
        raise UserGenerationError(
            "user generation failed semantic validation after regeneration: " + "; ".join(errors)
        )


def _generation_errors(
    result: UserGenerationResult,
    selected_ids: list[str],
    mode: RealizationMode,
) -> list[str]:
    errors: list[str] = []
    if result.selected_node_ids != selected_ids:
        errors.append(
            f"selected_node_ids must be exactly {selected_ids}, got {result.selected_node_ids}"
        )
    if result.realization_mode != mode:
        errors.append(f"realization_mode must be {mode.value}")
    if set(result.coverage_check) != set(selected_ids):
        errors.append("coverage_check keys must exactly match selected nodes")
    elif not all(result.coverage_check.values()):
        errors.append("all selected nodes must have true coverage")
    if result.contains_unsupported_intent:
        errors.append("contains_unsupported_intent must be false")
    if not result.user_message.strip():
        errors.append("user_message must be non-empty")
    return errors
