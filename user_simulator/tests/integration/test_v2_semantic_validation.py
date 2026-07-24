from user_simulator.config import GenerationSettings
from user_simulator.domain.dag import DagNode, Sample
from user_simulator.domain.enums import RealizationMode, SatisfactionLevel
from user_simulator.domain.state import EpisodeState
from user_simulator.llm.mock import MockStructuredLLMClient
from user_simulator.realizer.llm_realizer import LLMUserRealizer
from user_simulator.satisfaction.llm_updater import LLMSatisfactionUpdater


async def test_user_generation_retries_coverage_order_and_unsupported_intent() -> None:
    client = MockStructuredLLMClient(
        [
            {
                "user_message": "bad",
                "selected_node_ids": ["N2"],
                "realization_mode": "clear",
                "coverage": [{"node_id": "N3", "covered": True}],
                "contains_unsupported_intent": True,
                "summary": "invalid self-check",
            },
            {
                "user_message": "Could you finish the remaining budget comparison?",
                "selected_node_ids": ["N2"],
                "realization_mode": "clear",
                "coverage": [{"node_id": "N2", "covered": True}],
                "contains_unsupported_intent": False,
                "summary": "corrected without leakage",
            },
        ]
    )
    generation = GenerationSettings(temperature=0, max_completion_tokens=100)
    realizer = LLMUserRealizer(client, generation, generation)
    node = DagNode(
        node_id="N2",
        node_type="intent",
        node_intent="Finish the budget comparison.",
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
        unselected_unresolved_nodes=[],
        satisfaction={"N2": SatisfactionLevel.PARTIALLY_SATISFIED},
        history=[],
        latest_assistant_response="I compared quality but not price.",
        mode=RealizationMode.CLEAR,
        sample=sample,
    )
    assert result.coverage[0].node_id == "N2"
    assert not result.contains_unsupported_intent
    assert len(client.calls) == 2
    assert "Coverage IDs" in client.calls[1]["messages"][-1]["content"]
    assert "unsupported intent" in client.calls[1]["messages"][-1]["content"]


async def test_satisfaction_retries_generic_reason() -> None:
    client = MockStructuredLLMClient(
        [
            {
                "updates": [{"node_id": "N1", "status": "unsatisfied", "reason": "no"}],
                "summary": "generic",
            },
            {
                "updates": [
                    {
                        "node_id": "N1",
                        "status": "unsatisfied",
                        "reason": "The assistant only repeated the request and supplied no help.",
                    }
                ],
                "summary": "user disclosure is not assistant evidence",
            },
        ]
    )
    updater = LLMSatisfactionUpdater(
        client, GenerationSettings(temperature=0, max_completion_tokens=100)
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
