import json
from typing import cast

from user_simulator.config import GenerationSettings
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.domain.enums import RealizationMode, SatisfactionLevel
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import UserGenerationResult
from user_simulator.exceptions import StructuredOutputError, UserGenerationError
from user_simulator.llm.base import StructuredLLMClient
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.llm.schemas import UserGenerationResultV2
from user_simulator.realizer.base import UserRealizer


class LLMUserRealizer(UserRealizer):
    def __init__(
        self,
        client: StructuredLLMClient,
        clear_generation: GenerationSettings,
        abstract_generation: GenerationSettings,
        *,
        clear_prompt: PromptTemplate | None = None,
        abstract_prompt: PromptTemplate | None = None,
        clear_prompt_path: str = "configs/prompts/user_clear_v2.yaml",
        abstract_prompt_path: str = "configs/prompts/user_abstract_v2.yaml",
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
            RealizationMode.CLEAR: clear_prompt or PromptTemplate.load(clear_prompt_path),
            RealizationMode.ABSTRACT: abstract_prompt or PromptTemplate.load(abstract_prompt_path),
        }
        self.model_profile_names = {
            RealizationMode.CLEAR: model_profile_name,
            RealizationMode.ABSTRACT: abstract_model_profile_name or model_profile_name,
        }
        self.last_call_metadata: dict = {}
        self.semantic_events: list[dict] = []

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
        context = _labeled_context(
            [
                (
                    "SELECTED NODE DETAILS",
                    [item.model_dump() for item in selected_nodes],
                ),
                (
                    "SELECTED NODE SATISFACTION STATES",
                    {item.node_id: satisfaction[item.node_id].value for item in selected_nodes},
                ),
                (
                    "UNSELECTED UNRESOLVED NODE DETAILS (DO NOT EXPRESS)",
                    [item.model_dump() for item in unselected_unresolved_nodes],
                ),
                ("VISIBLE CONVERSATION", [item.model_dump() for item in history]),
                (
                    "LATEST ASSISTANT RESPONSE",
                    latest_assistant_response
                    if latest_assistant_response is not None
                    else "<INITIAL TURN: NO PREVIOUS ASSISTANT RESPONSE>",
                ),
                ("REALIZATION MODE", mode.value),
                ("TASK SUMMARY", sample.task_summary),
                ("TASK EXPECTATION", sample.task_expectation),
            ]
        )
        correction = ""
        self.semantic_events = []
        for attempt in range(2):
            prompt = self.prompts[mode]
            messages, metadata = prompt.render(
                context=context,
                selected_node_order=json.dumps(selected_ids),
                correction=correction,
            )
            metadata["model_profile"] = self.model_profile_names[mode]
            selected_client = self.clients[mode]
            try:
                result = cast(
                    UserGenerationResultV2,
                    await selected_client.generate_structured(
                        messages=messages,
                        response_model=prompt.schema.model,
                        schema_name=prompt.schema.name,
                        schema_version=prompt.schema.version,
                        generation=self.generations[mode],
                        prompt_metadata={**metadata, "semantic_retry_count": attempt},
                    ),
                )
            except StructuredOutputError as exc:
                raise UserGenerationError(str(exc)) from exc
            call_metadata = getattr(selected_client, "last_call_metadata", {})
            self.last_call_metadata = dict(call_metadata) if isinstance(call_metadata, dict) else {}
            errors = _generation_errors(result, selected_ids, mode)
            if not errors:
                self.semantic_events.append(
                    {
                        "semantic_attempt": attempt + 1,
                        "semantic_error": None,
                        "corrective_message": correction,
                    }
                )
                return result
            semantic_error = "; ".join(errors)
            self.semantic_events.append(
                {
                    "semantic_attempt": attempt + 1,
                    "semantic_error": semantic_error,
                    "corrective_message": correction,
                }
            )
            correction = (
                f"Correction required after invalid self-check: {semantic_error}. "
                "Regenerate the whole result."
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
    coverage_ids = [item.node_id for item in result.coverage]
    if coverage_ids != selected_ids or len(coverage_ids) != len(set(coverage_ids)):
        errors.append(f"Coverage IDs must exactly match selected IDs {selected_ids}")
    elif not all(item.covered for item in result.coverage):
        errors.append("all selected nodes must have true coverage")
    if result.contains_unsupported_intent:
        errors.append(
            "The message self-check reports an unsupported intent. Regenerate "
            f"using only {selected_ids}"
        )
    if not result.user_message.strip():
        errors.append("user_message must be non-empty")
    return errors


def _labeled_context(sections: list[tuple[str, object]]) -> str:
    return "\n\n".join(
        f"{label}\n{json.dumps(value, ensure_ascii=False, indent=2)}" for label, value in sections
    )
