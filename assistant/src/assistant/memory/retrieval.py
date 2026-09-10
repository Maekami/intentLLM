from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter, OrderedDict
from typing import Any, Protocol

from assistant.config import MemoryRetrievalSettings
from assistant.memory.models import MemoryEntry, RetrievalResult

_LATIN_TOKEN = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_EMBEDDING_CACHE: dict[tuple[str, str | None], dict[str, list[float]]] = {}
_EMBEDDING_CACHE_LOCK = threading.Lock()
_ENCODING_LOCKS: dict[tuple[str, str | None], threading.Lock] = {}
_INDEX_CACHE: OrderedDict[tuple[str, str | None, str], Any] = OrderedDict()
_INDEX_CACHE_LOCK = threading.Lock()
_MAX_CACHED_SNAPSHOT_INDEXES = 4


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
    """Embedding retrieval matching the official BGE/cosine reference path.

    Assistant components are rebuilt for every pipeline sample. The model,
    text embeddings, and a small LRU of normalized snapshot matrices therefore
    live at process scope so all samples in an evolution mini-batch reuse the
    same frozen-memory index.
    """

    def __init__(self, *, model_name: str, device: str | None) -> None:
        self.model_name = model_name
        self.device = device

    @property
    def cache_key(self) -> tuple[str, str | None]:
        return self.model_name, self.device

    @property
    def model(self) -> Any:
        key = self.cache_key
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
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode uncached texts together and preserve caller order."""

        if not texts:
            return []
        key = self.cache_key
        with _EMBEDDING_CACHE_LOCK:
            cache = _EMBEDDING_CACHE.setdefault(key, {})
            missing = list(dict.fromkeys(text for text in texts if text not in cache))
        if missing:
            encoding_lock = _encoding_lock(key)
            with encoding_lock:
                # Another retriever may have populated the process-wide cache
                # while this caller was waiting for the model.
                with _EMBEDDING_CACHE_LOCK:
                    cache = _EMBEDDING_CACHE.setdefault(key, {})
                    missing = [text for text in missing if text not in cache]
                if missing:
                    encoded = self.model.encode(missing, convert_to_numpy=True)
                    rows = encoded.tolist()
                    if len(rows) != len(missing):
                        raise ValueError(
                            "sentence transformer returned an unexpected embedding batch size"
                        )
                    additions = {
                        text: [float(value) for value in row]
                        for text, row in zip(missing, rows, strict=True)
                    }
                    with _EMBEDDING_CACHE_LOCK:
                        cache = _EMBEDDING_CACHE.setdefault(key, {})
                        cache.update(additions)
        with _EMBEDDING_CACHE_LOCK:
            cache = _EMBEDDING_CACHE[key]
            return [cache[text] for text in texts]

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
        entry_texts = [entry.to_text() for entry in entries]
        document_index = self._document_index(entry_texts)
        scores = _cosine_scores(query_embedding, document_index)
        scored = list(zip(entries, scores, strict=True))
        scored = [item for item in scored if item[1] >= min_score]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(entry=entry, score=score, rank=index + 1)
            for index, (entry, score) in enumerate(scored[:top_k])
        ]

    def _document_index(self, texts: list[str]) -> Any:
        index_key = (*self.cache_key, _snapshot_digest(texts))
        with _INDEX_CACHE_LOCK:
            cached = _INDEX_CACHE.get(index_key)
            if cached is not None:
                _INDEX_CACHE.move_to_end(index_key)
                return cached

        # NumPy is already a transitive dependency of sentence-transformers,
        # but remains optional for the dependency-free BM25 code path.
        import numpy as np

        matrix = np.asarray(self.encode_batch(texts), dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        normalized = np.zeros_like(matrix)
        nonzero = norms != 0.0
        normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
        with _INDEX_CACHE_LOCK:
            existing = _INDEX_CACHE.get(index_key)
            if existing is not None:
                _INDEX_CACHE.move_to_end(index_key)
                return existing
            _INDEX_CACHE[index_key] = normalized
            while len(_INDEX_CACHE) > _MAX_CACHED_SNAPSHOT_INDEXES:
                _INDEX_CACHE.popitem(last=False)
        return normalized


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


def _encoding_lock(key: tuple[str, str | None]) -> threading.Lock:
    with _MODEL_CACHE_LOCK:
        return _ENCODING_LOCKS.setdefault(key, threading.Lock())


def _snapshot_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cosine_scores(query: list[float], normalized_documents: Any) -> list[float]:
    import numpy as np

    query_array = np.asarray(query, dtype=np.float64)
    if normalized_documents.ndim != 2 or normalized_documents.shape[1] != len(query_array):
        raise ValueError("embedding vectors have different dimensions")
    query_norm = float(np.linalg.norm(query_array))
    if query_norm == 0.0:
        return [0.0] * int(normalized_documents.shape[0])
    scores = normalized_documents @ (query_array / query_norm)
    return [float(score) for score in scores.tolist()]
