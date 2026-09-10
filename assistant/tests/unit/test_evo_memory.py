from __future__ import annotations

from pathlib import Path

import pytest

import assistant.memory.retrieval as retrieval_module
from assistant.baselines.base import BaseBaseline
from assistant.config import GenerationSettings, MemorySettings, load_config, load_model_profile
from assistant.domain.messages import ChatMessage
from assistant.factory import build_assistant_components
from assistant.llm.base import GeneratedResponse
from assistant.memory.models import MemoryEntry, RetrievalResult
from assistant.memory.retrieval import SentenceTransformerRetriever, build_retriever
from assistant.memory.sessions import (
    ExpRAGSession,
    ReMemSession,
    _build_retrieval_query,
    _parse_remem_action,
)
from assistant.memory.store import JsonMemoryStore, MemoryStoreError


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


class RecordingRetriever:
    def __init__(self, task_ids_by_call: list[list[str]]) -> None:
        self.task_ids_by_call = task_ids_by_call
        self.calls: list[dict] = []

    def retrieve(self, query, entries, *, top_k, min_score):
        self.calls.append(
            {
                "query": query,
                "entry_ids": [entry.task_id for entry in entries],
                "top_k": top_k,
                "min_score": min_score,
            }
        )
        selected_ids = self.task_ids_by_call[len(self.calls) - 1]
        entries_by_id = {entry.task_id: entry for entry in entries}
        return [
            RetrievalResult(entry=entries_by_id[task_id], score=1.0, rank=index)
            for index, task_id in enumerate(selected_ids, 1)
        ]


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


def test_per_turn_retrieval_query_prioritizes_latest_visible_user_context() -> None:
    query = _build_retrieval_query(
        [
            ChatMessage(role="user", content="Earlier broad request " + "x" * 400),
            ChatMessage(role="assistant", content="assistant text must not enter retrieval"),
            ChatMessage(role="user", content="LATEST budget and date constraints"),
        ],
        maximum=256,
    )

    assert query.startswith("CURRENT USER REQUEST:\nLATEST budget and date constraints")
    assert len(query) == 256
    assert "assistant text must not enter retrieval" not in query


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


async def test_exprag_retrieves_each_turn_from_one_frozen_episode_bank(tmp_path) -> None:
    settings = memory_settings(tmp_path, "exprag")
    store = JsonMemoryStore(settings)
    broad = MemoryEntry(
        task_id="broad-trip",
        input_text="Plan a general holiday",
        output_text="Start with dates and destination.",
        is_successful=True,
    )
    budget = MemoryEntry(
        task_id="budget-trip",
        input_text="Plan a low-cost holiday",
        output_text="Reserve a hard contingency budget.",
        is_successful=True,
    )
    retriever = RecordingRetriever([["broad-trip"], ["budget-trip"]])
    client = FakeMemoryClient(["Where would you like to go?", "Use the budget example."])
    session = ExpRAGSession(
        client,
        GenerationSettings(max_completion_tokens=100),
        BaseBaseline(),
        settings=settings,
        store=store,
        retriever=retriever,
        memory_snapshot=(broad, budget),
    )

    await session.respond("Help me plan a holiday")
    # A write after the episode starts must not change its frozen memory bank.
    store.upsert(
        MemoryEntry(
            task_id="same-batch-late-entry",
            input_text="Must remain invisible",
            output_text="Must remain invisible",
            is_successful=True,
        )
    )
    await session.respond("Keep the total budget under 500 euros")

    assert len(retriever.calls) == 2
    assert retriever.calls[0]["query"] == "Help me plan a holiday"
    second_query = retriever.calls[1]["query"]
    assert second_query.index("Keep the total budget") < second_query.index(
        "Help me plan a holiday"
    )
    assert retriever.calls[0]["entry_ids"] == ["broad-trip", "budget-trip"]
    assert retriever.calls[1]["entry_ids"] == ["broad-trip", "budget-trip"]
    second_prompt = client.calls[1][-1]["content"]
    assert "Reserve a hard contingency budget" in second_prompt
    assert "Start with dates and destination" not in second_prompt
    assert session.last_call_metadata["memory_retrieval_scope"] == "per_turn"
    assert session.last_call_metadata["memory_bank_scope"] == "frozen_episode_snapshot"
    assert session.last_call_metadata["memory_bank_snapshot_source"] == "provided_snapshot"
    assert session.last_call_metadata["memory_bank_entry_count"] == 2
    assert session.last_call_metadata["memory_retrieved_task_ids"] == ["budget-trip"]
    assert len(session.last_call_metadata["memory_retrieval_query_sha256"]) == 64


async def test_prepared_update_waits_for_atomic_batch_commit(tmp_path) -> None:
    settings = memory_settings(tmp_path, "exprag")
    store = JsonMemoryStore(settings)
    session = ExpRAGSession(
        FakeMemoryClient(["A deferred answer."]),
        GenerationSettings(max_completion_tokens=100),
        BaseBaseline(),
        settings=settings,
        store=store,
        retriever=build_retriever(settings.retrieval),
        memory_snapshot=(),
    )

    await session.respond("A development task")
    snapshot = store.entries()
    prepared = session.prepare_memory_update(task_id="dev-1", success=True)

    assert prepared.ready is True
    assert store.entries() == []
    results = store.upsert_many([prepared.entry], expected_entries=snapshot)
    assert [result.stored for result in results] == [True]
    assert [entry.task_id for entry in store.entries()] == ["dev-1"]


def test_batch_commit_rejects_a_changed_snapshot(tmp_path) -> None:
    settings = memory_settings(tmp_path, "exprag")
    store = JsonMemoryStore(settings)
    snapshot = store.entries()
    store.upsert(
        MemoryEntry(task_id="external", input_text="x", output_text="y", is_successful=True)
    )

    with pytest.raises(MemoryStoreError, match="changed while an evolution mini-batch"):
        store.upsert_many(
            [MemoryEntry(task_id="dev", input_text="a", output_text="b", is_successful=True)],
            expected_entries=snapshot,
        )


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
    assert metadata["memory_retrieved_count"] == 2
    assert metadata["memory_retained_count"] == 1
    assert metadata["memory_pruned_task_ids"] == ["irrelevant-memory"]


async def test_remem_retrieves_once_per_turn_and_resets_pruned_working_set(tmp_path) -> None:
    settings = memory_settings(tmp_path, "remem")
    store = JsonMemoryStore(settings)
    memory = MemoryEntry(
        task_id="trip-memory",
        input_text="Plan a trip within a fixed budget",
        output_text="Track accommodation and transport separately.",
        is_successful=True,
    )
    retriever = RecordingRetriever([["trip-memory"], ["trip-memory"]])
    client = FakeMemoryClient(
        [
            "Think-Prune: 1",
            "Final Answer: Which destination do you prefer?",
            "Final Answer: Reuse the relevant budget strategy.",
        ]
    )
    session = ReMemSession(
        client,
        GenerationSettings(max_completion_tokens=100),
        BaseBaseline(),
        settings=settings,
        store=store,
        retriever=retriever,
        memory_snapshot=(memory,),
    )

    assert await session.respond("Help me plan a trip") == "Which destination do you prefer?"
    first_metadata = session.last_call_metadata
    assert first_metadata["internal_call_count"] == 2
    assert first_metadata["memory_retrieved_task_ids"] == ["trip-memory"]
    assert first_metadata["memory_retained_task_ids"] == []
    assert first_metadata["memory_pruned_task_ids"] == ["trip-memory"]
    assert session.retrieved == ()

    assert await session.respond("It must stay under 500 euros") == (
        "Reuse the relevant budget strategy."
    )
    assert len(retriever.calls) == 2
    assert len(client.calls) == 3
    assert "[Memory 1]" not in client.calls[1][-1]["content"]
    assert "[Memory 1]" in client.calls[2][-1]["content"]
    assert retriever.calls[1]["query"].index("It must stay under 500 euros") < (
        retriever.calls[1]["query"].index("Help me plan a trip")
    )
    assert session.last_call_metadata["internal_call_count"] == 1
    assert session.last_call_metadata["memory_retrieved_task_ids"] == ["trip-memory"]
    assert session.last_call_metadata["memory_retained_task_ids"] == ["trip-memory"]
    assert session.last_call_metadata["memory_pruned_task_ids"] == []


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("- Think: inspect the retrieved memories", ("think", "inspect the retrieved memories")),
        ("* Think-Prune: 1,3", ("refine", "1,3")),
        ("+ Final Answer: a clean answer", ("act", "a clean answer")),
        ("- A normal bulleted answer", ("act", "- A normal bulleted answer")),
    ],
)
def test_remem_parser_normalizes_markers_only_before_control_labels(
    response: str,
    expected: tuple[str, str],
) -> None:
    assert _parse_remem_action(response) == expected


async def test_remem_prunes_only_the_control_line_from_a_bulleted_response(
    tmp_path,
) -> None:
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
    client = FakeMemoryClient(
        [
            "- Think: inspect which memory is useful.",
            "- Think-Prune: 2\n- Final Answer: this embedded answer must be ignored.",
            "+ Final Answer: Take the direct train.",
        ]
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

    assert result == "Take the direct train."
    assert len(client.calls) == 3
    assert "[Memory 2]" in client.calls[1][-1]["content"]
    assert "[Memory 2]" not in client.calls[2][-1]["content"]
    assert session.last_call_metadata["remem_operation_counts"] == {
        "think": 1,
        "refine": 1,
        "act": 1,
    }
    assert "You MUST respond in EXACTLY ONE of these formats." in (client.calls[0][-1]["content"])
    assert "\n- Think:" not in client.calls[0][-1]["content"]
    assert "\n- Think:" not in client.calls[0][0]["content"]


def test_model_profiles_enable_memory_without_changing_base_profiles() -> None:
    conditions = {
        "qwen_3_6_27b_vllm_non_thinking": (
            "qwen_3_6_27b_exprag",
            "qwen_3_6_27b_remem",
        ),
        "gpt_5_6_luna_non_thinking": (
            "gpt_5_6_luna_exprag",
            "gpt_5_6_luna_remem",
        ),
        "gemini_3_6_flash_non_thinking": (
            "gemini_3_6_flash_exprag",
            "gemini_3_6_flash_remem",
        ),
    }
    memory_paths = set()

    for base_name, memory_names in conditions.items():
        original = load_model_profile(base_name)
        assert original.memory is None
        for framework, memory_name in zip(("exprag", "remem"), memory_names, strict=True):
            profile = load_model_profile(memory_name)
            assert profile.profile_name == memory_name
            assert profile.provider == original.provider
            assert profile.model_id == original.model_id
            assert profile.base_url == original.base_url
            assert profile.reasoning == original.reasoning
            assert profile.generation == original.generation
            assert profile.memory.framework == framework
            assert profile.memory.max_entries == 2000
            assert profile.memory.path.is_absolute()
            assert profile.memory.retrieval.backend == "sentence_transformers"
            assert profile.memory.retrieval.embedding_model == "BAAI/bge-base-en-v1.5"
            assert profile.memory.retrieval.top_k == 4
            assert profile.memory.retrieval.min_score == 0.0
            assert profile.memory.retrieval.query_max_characters == 2048
            assert profile.memory.retrieval.device == "cpu"
            memory_paths.add(profile.memory.path)

    assert len(memory_paths) == 6


def test_bge_retrieval_batches_and_reuses_snapshot_embeddings(monkeypatch) -> None:
    np = pytest.importorskip("numpy")

    class FakeEmbeddingModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts, *, convert_to_numpy):
            assert convert_to_numpy is True
            values = list(texts)
            self.calls.append(values)
            vectors = []
            for text in values:
                lowered = text.lower()
                if "alpha" in lowered:
                    vectors.append([1.0, 0.0])
                elif "beta" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([1.0, 1.0])
            return np.asarray(vectors)

    key = ("fake-bge", "cpu")
    model = FakeEmbeddingModel()
    monkeypatch.setattr(retrieval_module, "_MODEL_CACHE", {key: model})
    monkeypatch.setattr(retrieval_module, "_EMBEDDING_CACHE", {})
    monkeypatch.setattr(retrieval_module, "_ENCODING_LOCKS", {})
    monkeypatch.setattr(retrieval_module, "_INDEX_CACHE", retrieval_module.OrderedDict())
    entries = [
        MemoryEntry(task_id="alpha", input_text="Alpha task", output_text="Alpha answer"),
        MemoryEntry(task_id="beta", input_text="Beta task", output_text="Beta answer"),
    ]

    first = SentenceTransformerRetriever(model_name=key[0], device=key[1])
    assert [
        item.entry.task_id
        for item in first.retrieve("alpha query", entries, top_k=2, min_score=0.0)
    ] == ["alpha", "beta"]
    # Query is one batch; both previously unseen memory entries are another.
    assert [len(call) for call in model.calls] == [1, 2]

    second = SentenceTransformerRetriever(model_name=key[0], device=key[1])
    assert [
        item.entry.task_id
        for item in second.retrieve("beta query", entries, top_k=2, min_score=0.0)
    ] == ["beta", "alpha"]
    # A fresh retriever only encodes the new query. It reuses both the text
    # embeddings and normalized index for the identical frozen snapshot.
    assert [len(call) for call in model.calls] == [1, 2, 1]


def test_factory_selects_memory_session_from_model_profile() -> None:
    config = load_config(cli_overrides={"models": {"assistant": "qwen_3_6_27b_exprag"}})
    client = FakeMemoryClient(["answer"])
    components = build_assistant_components(config=config, client=client)

    assert isinstance(components.session, ExpRAGSession)
    assert components.memory_framework == "exprag"
