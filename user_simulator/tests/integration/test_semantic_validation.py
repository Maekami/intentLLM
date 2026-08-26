from user_simulator.config import GenerationSettings
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.domain.enums import RealizationMode, SatisfactionLevel
from user_simulator.domain.state import EpisodeState
from user_simulator.llm.mock import MockStructuredLLMClient
from user_simulator.llm.prompt import PromptTemplate
from user_simulator.realizer.llm_realizer import LLMUserRealizer
from user_simulator.satisfaction.llm_updater import LLMSatisfactionUpdater


async def test_user_generation_retries_coverage_order_and_unsupported_task_content() -> None:
    client = MockStructuredLLMClient(
        [
            {
                "user_message": "bad",
                "selected_node_ids": ["N2"],
                "realization_mode": "clear",
                "coverage": [{"node_id": "N3", "covered": True}],
                "contains_unsupported_task_content": True,
                "summary": "invalid self-check",
            },
            {
                "user_message": "Could you finish the remaining budget comparison?",
                "selected_node_ids": ["N2"],
                "realization_mode": "clear",
                "coverage": [{"node_id": "N2", "covered": True}],
                "contains_unsupported_task_content": False,
                "summary": "corrected without leakage",
            },
        ]
    )
    generation = GenerationSettings(temperature=0, max_completion_tokens=100)
    realizer = LLMUserRealizer(
        client,
        generation,
        generation,
        clear_prompt=PromptTemplate.load("configs/prompts/user_clear.yaml"),
        abstract_prompt=PromptTemplate.load("configs/prompts/user_abstract.yaml"),
    )
    node = DagNode(
        node_id="N2",
        node_type="intent",
        node_intent="Finish the budget comparison.",
    )
    unselected = DagNode(
        node_id="N3",
        node_type="intent",
        node_intent="Reveal the user's private movie hobby.",
    )
    sample = Sample.model_validate(
        {
            "sample_id": "x",
            "reason_dag": {
                "nodes": [
                    node.model_dump(),
                    {
                        "node_id": "END",
                        "node_type": "terminal",
                        "node_intent": "done",
                    },
                ],
                "edges": [{"edge_id": "E", "source": "N2", "target": "END"}],
            },
        }
    )
    result = await realizer.generate(
        selected_nodes=[node],
        unselected_unresolved_nodes=[unselected],
        satisfaction={"N2": SatisfactionLevel.PARTIALLY_SATISFIED},
        selected_remaining_gaps={
            "N2": "Compare the remaining budget implications.",
            "N3": "This unselected gap must not appear.",
        },
        history=[],
        latest_assistant_response="I compared quality but not price.",
        mode=RealizationMode.CLEAR,
        sample=sample,
    )
    assert result.coverage[0].node_id == "N2"
    assert not result.contains_unsupported_task_content
    assert len(client.calls) == 2
    assert "Coverage IDs" in client.calls[1]["messages"][-1]["content"]
    assert "unsupported-task-content" in client.calls[1]["messages"][-1]["content"]
    rendered = client.calls[0]["messages"][-1]["content"]
    for section in (
        "TURN TYPE",
        "TASK SUMMARY — DIRECTION ONLY",
        "TASK EXPECTATION — DIRECTION ONLY",
        "VISIBLE CONVERSATION",
        "LATEST ASSISTANT RESPONSE",
        "SELECTED NODE DETAILS",
        "SELECTED NODE SATISFACTION STATES",
        "SELECTED NODE REMAINING GAPS",
        "UNSELECTED UNRESOLVED NODE IDS — DO NOT EXPRESS",
        "REALIZATION MODE",
    ):
        assert section in rendered
    assert '"follow_up"' in rendered
    assert "I compared quality but not price." in rendered
    assert "Compare the remaining budget implications." in rendered
    assert "This unselected gap must not appear." not in rendered
    assert '"N3"' in rendered
    assert "Reveal the user's private movie hobby." not in rendered


async def test_user_generation_uses_configured_third_semantic_attempt() -> None:
    client = MockStructuredLLMClient(
        [
            {
                "user_message": "I still need the next step.",
                "selected_node_ids": ["N1"],
                "realization_mode": "clear",
                "coverage": [{"node_id": "N9", "covered": True}],
                "contains_unsupported_task_content": False,
                "summary": "wrong coverage ID",
            },
            {
                "user_message": "What should I do next?",
                "selected_node_ids": ["N1"],
                "realization_mode": "clear",
                "coverage": [{"node_id": "N8", "covered": True}],
                "contains_unsupported_task_content": False,
                "summary": "still wrong coverage ID",
            },
            {
                "user_message": "What should I do next?",
                "selected_node_ids": ["N1"],
                "realization_mode": "clear",
                "coverage": [{"node_id": "N1", "covered": True}],
                "contains_unsupported_task_content": False,
                "summary": "correct coverage ID",
            },
        ]
    )
    generation = GenerationSettings(temperature=0, max_completion_tokens=100)
    realizer = LLMUserRealizer(
        client,
        generation,
        generation,
        clear_prompt=PromptTemplate.load("configs/prompts/user_clear.yaml"),
        abstract_prompt=PromptTemplate.load("configs/prompts/user_abstract.yaml"),
        semantic_attempts=3,
        abstract_semantic_attempts=2,
    )
    node = DagNode(
        node_id="N1",
        node_type="intent",
        node_intent="Ask for the next step.",
    )
    sample = Sample.model_validate(
        {
            "sample_id": "third-attempt",
            "reason_dag": {
                "nodes": [
                    node.model_dump(),
                    {"node_id": "END", "node_type": "terminal", "node_intent": "done"},
                ],
                "edges": [{"edge_id": "E", "source": "N1", "target": "END"}],
            },
        }
    )

    result = await realizer.generate(
        selected_nodes=[node],
        unselected_unresolved_nodes=[],
        satisfaction={"N1": SatisfactionLevel.UNSATISFIED},
        selected_remaining_gaps={},
        history=[],
        latest_assistant_response="Start with the first step.",
        mode=RealizationMode.CLEAR,
        sample=sample,
    )

    assert result.coverage[0].node_id == "N1"
    assert len(client.calls) == 3
    assert [call["prompt_metadata"]["semantic_retry_count"] for call in client.calls] == [0, 1, 2]
    expected = '`coverage` must be exactly [{"node_id": "N1", "covered": true}]'
    assert expected in client.calls[1]["messages"][-1]["content"]
    assert expected in client.calls[2]["messages"][-1]["content"]


async def test_satisfaction_retries_generic_reason() -> None:
    client = MockStructuredLLMClient(
        [
            {
                "updates": [
                    {
                        "node_id": "N1",
                        "status": "unsatisfied",
                        "reason": "no",
                        "remaining_gap": None,
                    }
                ],
                "summary": "generic",
            },
            {
                "updates": [
                    {
                        "node_id": "N1",
                        "status": "unsatisfied",
                        "reason": "The assistant only repeated the request and supplied no help.",
                        "remaining_gap": None,
                    }
                ],
                "summary": "user disclosure is not assistant evidence",
            },
        ]
    )
    updater = LLMSatisfactionUpdater(
        client,
        GenerationSettings(temperature=0, max_completion_tokens=100),
        prompt=PromptTemplate.load("configs/prompts/satisfaction.yaml"),
    )
    state = EpisodeState(
        sample_id="x",
        difficulty="easy",
        current_frontier="N1",
        exposed_nodes=["N1"],
        satisfaction={"N1": SatisfactionLevel.UNSATISFIED},
        random_seed=1,
    )
    node = DagNode(
        node_id="N1",
        node_type="intent",
        node_intent="Obtain actionable help.",
    )
    result = await updater.update(
        history=[],
        latest_assistant_response="What kind of help?",
        state=state,
        exposed_nodes=[node],
    )
    assert result.updates[0].status == SatisfactionLevel.UNSATISFIED
    assert len(client.calls) == 2
    assert "assistant-provided evidence" in client.calls[1]["messages"][-1]["content"]
    rendered_context = client.calls[0]["messages"][-1]["content"]
    assert "EXPOSED NODE DETAILS IN REQUIRED ORDER" in rendered_context
    assert "VISIBLE CONVERSATION WITH ROLES" in rendered_context
    assert "TASK SUMMARY" not in rendered_context
    assert "TASK EXPECTATION" not in rendered_context
