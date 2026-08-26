import json

from user_simulator.config import GenerationSettings
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.domain.enums import RealizationMode, SatisfactionLevel
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import UserGenerationResult
from user_simulator.exceptions import StructuredOutputError, UserGenerationError
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
        clear_prompt: PromptTemplate,
        abstract_prompt: PromptTemplate,
        model_profile_name: str = "deepseek_v4_flash_0731",
        abstract_client: StructuredLLMClient | None = None,
        abstract_model_profile_name: str | None = None,
        semantic_attempts: int = 3,
        abstract_semantic_attempts: int | None = None,
    ) -> None:
        if semantic_attempts < 1:
            raise ValueError("semantic_attempts must be at least 1")
        if abstract_semantic_attempts is not None and abstract_semantic_attempts < 1:
            raise ValueError("abstract_semantic_attempts must be at least 1")
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
            RealizationMode.CLEAR: clear_prompt,
            RealizationMode.ABSTRACT: abstract_prompt,
        }
        self.model_profile_names = {
            RealizationMode.CLEAR: model_profile_name,
            RealizationMode.ABSTRACT: abstract_model_profile_name or model_profile_name,
        }
        self.semantic_attempts = {
            RealizationMode.CLEAR: semantic_attempts,
            RealizationMode.ABSTRACT: abstract_semantic_attempts or semantic_attempts,
        }
        self.last_call_metadata: dict = {}
        self.semantic_events: list[dict] = []

    async def generate(
        self,
        *,
        selected_nodes: list[DagNode],
        unselected_unresolved_nodes: list[DagNode],
        satisfaction: dict[str, SatisfactionLevel],
        selected_remaining_gaps: dict[str, str],
        history: list[ChatMessage],
        latest_assistant_response: str | None,
        mode: RealizationMode,
        sample: Sample,
    ) -> UserGenerationResult:
        selected_ids = [item.node_id for item in selected_nodes]
        context = realizer_context(
            selected_nodes=selected_nodes,
            unselected_unresolved_nodes=unselected_unresolved_nodes,
            satisfaction=satisfaction,
            selected_remaining_gaps=selected_remaining_gaps,
            history=history,
            latest_assistant_response=latest_assistant_response,
            mode=mode,
            sample=sample,
        )
        correction = ""
        self.semantic_events = []
        attempt_limit = self.semantic_attempts[mode]
        for attempt in range(attempt_limit):
            prompt = self.prompts[mode]
            messages, metadata = prompt.render(
                context=context,
                selected_node_order=json.dumps(selected_ids),
                correction=correction,
            )
            metadata["model_profile"] = self.model_profile_names[mode]
            selected_client = self.clients[mode]
            try:
                result = await selected_client.generate_structured(
                    messages=messages,
                    response_model=UserGenerationResult,
                    schema_name=prompt.schema.name,
                    generation=self.generations[mode],
                    prompt_metadata={**metadata, "semantic_retry_count": attempt},
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
            required_coverage = [
                {"node_id": node_id, "covered": True} for node_id in selected_ids
            ]
            correction = (
                f"Correction required after invalid model-reported checks: {semantic_error}. "
                f"`selected_node_ids` must be exactly {json.dumps(selected_ids)}. "
                f"`coverage` must be exactly {json.dumps(required_coverage)}. "
                "Regenerate the whole result and preserve these exact checks."
            )
        raise UserGenerationError(
            f"user generation failed semantic validation after {attempt_limit} attempts: "
            + "; ".join(errors)
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
    if result.contains_unsupported_task_content:
        errors.append(
            "The model-reported unsupported-task-content check is true. Regenerate "
            "using only selected-node content and facts previously stated by the user: "
            f"{selected_ids}"
        )
    if not result.user_message.strip():
        errors.append("user_message must be non-empty")
    return errors


def realizer_context(
    *,
    selected_nodes: list[DagNode],
    unselected_unresolved_nodes: list[DagNode],
    satisfaction: dict[str, SatisfactionLevel],
    selected_remaining_gaps: dict[str, str],
    history: list[ChatMessage],
    latest_assistant_response: str | None,
    mode: RealizationMode,
    sample: Sample,
) -> str:
    turn_type = "initial" if latest_assistant_response is None else "follow_up"
    filtered_remaining_gaps = {
        item.node_id: selected_remaining_gaps[item.node_id]
        for item in selected_nodes
        if item.node_id in selected_remaining_gaps
    }
    return _labeled_context(
        [
            ("TURN TYPE", turn_type),
            ("TASK SUMMARY — DIRECTION ONLY", sample.task_summary),
            ("TASK EXPECTATION — DIRECTION ONLY", sample.task_expectation),
            ("VISIBLE CONVERSATION", [item.model_dump() for item in history]),
            (
                "LATEST ASSISTANT RESPONSE",
                latest_assistant_response if latest_assistant_response is not None else "none",
            ),
            (
                "SELECTED NODE DETAILS",
                [item.model_dump() for item in selected_nodes],
            ),
            (
                "SELECTED NODE SATISFACTION STATES",
                {item.node_id: satisfaction[item.node_id].value for item in selected_nodes},
            ),
            ("SELECTED NODE REMAINING GAPS", filtered_remaining_gaps),
            (
                "UNSELECTED UNRESOLVED NODE IDS — DO NOT EXPRESS",
                [item.node_id for item in unselected_unresolved_nodes],
            ),
            ("REALIZATION MODE", mode.value),
        ]
    )


def _labeled_context(sections: list[tuple[str, object]]) -> str:
    return "\n\n".join(
        f"{label}\n{json.dumps(value, ensure_ascii=False, indent=2)}" for label, value in sections
    )
