from __future__ import annotations

from pathlib import Path

from assistant.baselines.base import BaseBaseline
from assistant.config import GenerationSettings, MemorySettings, load_config, load_model_profile
from assistant.factory import build_assistant_components
from assistant.llm.base import GeneratedResponse
from assistant.memory.models import MemoryEntry
from assistant.memory.retrieval import build_retriever
from assistant.memory.sessions import ExpRAGSession, ReMemSession
from assistant.memory.store import JsonMemoryStore


class FakeMemoryClient:
    def __init__(
        self,
        responses: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.metadata = list(metadata or [{} for _ in responses])
        self.calls: list[list[dict]] = []
        self.last_call_metadata: dict = {}
        self.profile = type("Profile", (), {"profile_name": "fake-profile"})()

    async def generate(self, *, messages, generation):
        self.calls.append(messages)
        self.last_call_metadata = self.metadata.pop(0)
        return GeneratedResponse(content=self.responses.pop(0))


def memory_settings(tmp_path: Path, framework: str) -> MemorySettings:
    return MemorySettings.model_validate(
        {
            "framework": framework,
            "path": tmp_path / f"{framework}.json",
            "retrieval": {"backend": "bm25", "top_k": 4, "min_score": 0.0},
            "remem": {"max_iterations": 5, "enable_pruning": True},
        }
    )


def seed_memory(store: JsonMemoryStore) -> None:
    store.upsert(
        MemoryEntry(
            task_id="trip-memory",
            input_text="Plan a train trip through Spain",
            output_text="Reserve flexible rail tickets and group nearby stops.",
            feedback="The itinerary was accepted.",
            is_successful=True,
        )
    )


async def test_exprag_retrieves_context_and_evolves_after_finalization(tmp_path) -> None:
    settings = memory_settings(tmp_path, "exprag")
    store = JsonMemoryStore(settings)
    seed_memory(store)
    client = FakeMemoryClient(
        ["Use a flexible rail pass."],
        [{"input_tokens": 10, "output_tokens": 5}],
    )
    session = ExpRAGSession(
        client,
        GenerationSettings(max_completion_tokens=100),
        BaseBaseline(),
        settings=settings,
        store=store,
        retriever=build_retriever(settings.retrieval),
    )

    assert await session.respond("Help me plan a train trip in Spain") == (
        "Use a flexible rail pass."
    )
    prompt = client.calls[0][-1]["content"]
    assert "RELEVANT EXPERIENCE FROM SIMILAR TASKS" in prompt
    assert "Reserve flexible rail tickets" in prompt
    assert session.last_call_metadata["memory_framework"] == "exprag"
    update = session.finalize_task(
        task_id="new-task",
        success=True,
        feedback="Episode terminated normally.",
    )
    assert update["stored"] is True
    assert {entry.task_id for entry in store.entries()} == {"trip-memory", "new-task"}


async def test_remem_runs_think_prune_act_and_aggregates_all_usage(tmp_path) -> None:
    settings = memory_settings(tmp_path, "remem")
    store = JsonMemoryStore(settings)
    seed_memory(store)
    store.upsert(
        MemoryEntry(
            task_id="irrelevant-memory",
            input_text="Bake a loaf of bread",
            output_text="Use bread flour.",
            is_successful=True,
        )
    )
    per_call = [
        {
            "model_id": "fake",
            "input_tokens": 2,
            "output_tokens": 3,
            "thinking_tokens": 1,
            "answer_tokens": 2,
            "latency_seconds": 0.1,
            "transport_retry_count": 0,
        }
        for _ in range(3)
    ]
    client = FakeMemoryClient(
        [
            "Think: I should identify which experience is useful.",
            "Think-Prune: 2",
            "Final Answer: Take the direct train and keep the ticket flexible.",
        ],
        per_call,
    )
    session = ReMemSession(
        client,
        GenerationSettings(max_completion_tokens=100),
        BaseBaseline(),
        settings=settings,
        store=store,
        retriever=build_retriever(settings.retrieval),
    )

    result = await session.respond("Help me plan a train trip in Spain")
    assert result == "Take the direct train and keep the ticket flexible."
    assert len(client.calls) == 3
    assert "[Memory 2]" in client.calls[1][-1]["content"]
    assert "[Memory 2]" not in client.calls[2][-1]["content"]
    assert len(session.retrieved) == 1
    metadata = session.last_call_metadata
    assert metadata["internal_call_count"] == 3
    assert metadata["input_tokens"] == 6
    assert metadata["output_tokens"] == 9
    assert metadata["thinking_tokens"] == 3
    assert metadata["answer_tokens"] == 6
    assert metadata["remem_operation_counts"] == {"think": 1, "refine": 1, "act": 1}


def test_model_profiles_enable_memory_without_changing_base_profiles() -> None:
    original = load_model_profile("qwen_3_6_27b")
    exprag = load_model_profile("qwen_3_6_27b_exprag")
    remem = load_model_profile("qwen_3_6_27b_remem")

    assert original.memory is None
    assert exprag.memory.framework == "exprag"
    assert remem.memory.framework == "remem"
    assert exprag.memory.path.is_absolute()
    assert exprag.model_id == original.model_id == remem.model_id


def test_factory_selects_memory_session_from_model_profile() -> None:
    config = load_config(cli_overrides={"models": {"assistant": "qwen_3_6_27b_exprag"}})
    client = FakeMemoryClient(["answer"])
    components = build_assistant_components(config=config, client=client)

    assert isinstance(components.session, ExpRAGSession)
    assert components.memory_framework == "exprag"
