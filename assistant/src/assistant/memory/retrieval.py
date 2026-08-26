from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Any, Protocol

from assistant.config import MemoryRetrievalSettings
from assistant.memory.models import MemoryEntry, RetrievalResult

_LATIN_TOKEN = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class MemoryRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        entries: list[MemoryEntry],
        *,
        top_k: int,
        min_score: float,
    ) -> list[RetrievalResult]: ...


class BM25Retriever:
    """Dependency-free lexical fallback for source-only execution."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: str,
        entries: list[MemoryEntry],
        *,
        top_k: int,
        min_score: float,
    ) -> list[RetrievalResult]:
        if not entries:
            return []
        documents = [_tokens(entry.to_text()) for entry in entries]
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        average_length = sum(map(len, documents)) / max(len(documents), 1)
        document_frequency = Counter(token for document in documents for token in set(document))
        scored: list[tuple[MemoryEntry, float]] = []
        for entry, document in zip(entries, documents, strict=True):
            frequencies = Counter(document)
            score = 0.0
            for token, query_frequency in Counter(query_tokens).items():
                frequency = frequencies[token]
                if frequency == 0:
                    continue
                count = document_frequency[token]
                inverse_frequency = math.log(1.0 + (len(documents) - count + 0.5) / (count + 0.5))
                normalization = frequency + self.k1 * (
                    1.0 - self.b + self.b * len(document) / max(average_length, 1.0)
                )
                score += (
                    inverse_frequency
                    * frequency
                    * (self.k1 + 1.0)
                    / normalization
                    * query_frequency
                )
            if score >= min_score:
                scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(entry=entry, score=score, rank=index + 1)
            for index, (entry, score) in enumerate(scored[:top_k])
        ]


class SentenceTransformerRetriever:
    """Embedding retrieval matching the official BGE-based reference option."""

    def __init__(self, *, model_name: str, device: str | None) -> None:
        self.model_name = model_name
        self.device = device
        self._embedding_cache: dict[str, list[float]] = {}

    @property
    def model(self) -> Any:
        key = (self.model_name, self.device)
        with _MODEL_CACHE_LOCK:
            if key not in _MODEL_CACHE:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "retrieval.backend=sentence_transformers requires the optional "
                        "sentence-transformers package; use backend=bm25 for dependency-free "
                        "source execution"
                    ) from exc
                _MODEL_CACHE[key] = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
            return _MODEL_CACHE[key]

    def encode(self, text: str) -> list[float]:
        if text not in self._embedding_cache:
            encoded = self.model.encode(text, convert_to_numpy=True)
            self._embedding_cache[text] = [float(value) for value in encoded.tolist()]
        return self._embedding_cache[text]

    def retrieve(
        self,
        query: str,
        entries: list[MemoryEntry],
        *,
        top_k: int,
        min_score: float,
    ) -> list[RetrievalResult]:
        if not entries:
            return []
        query_embedding = self.encode(query)
        scored = [
            (entry, _cosine(query_embedding, self.encode(entry.to_text()))) for entry in entries
        ]
        scored = [item for item in scored if item[1] >= min_score]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(entry=entry, score=score, rank=index + 1)
            for index, (entry, score) in enumerate(scored[:top_k])
        ]


def build_retriever(settings: MemoryRetrievalSettings) -> MemoryRetriever:
    if settings.backend == "sentence_transformers":
        return SentenceTransformerRetriever(
            model_name=settings.embedding_model,
            device=settings.device,
        )
    return BM25Retriever(k1=settings.bm25_k1, b=settings.bm25_b)


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = _LATIN_TOKEN.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors have different dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
