"""Embedding providers.

- `LocalHashEmbedding`: deterministic, dependency-free, offline. Uses
  hashed token buckets with L2 normalization. Good for tests, demos,
  and CI without model downloads.
- `SentenceTransformerEmbedding`: lazy-loads sentence-transformers only
  when instantiated, so the rest of the app never pays the import cost.

Factory `get_embedding_provider()` reads Settings.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from src.config import Settings, get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two (already normalized) vectors."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of `dim`-length vectors."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]


class LocalHashEmbedding(EmbeddingProvider):
    """Deterministic hashed-bag-of-tokens embedding. Fully offline.

    Each alphanumeric token hashes into one of `dim` buckets (signed
    hashing à la feature hashing). Vectors are L2-normalized so cosine
    similarity behaves like lexical overlap with saturation.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
                vec[idx] += sign
            out.append(_normalize(vec))
        return out


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local sentence-transformers model. Lazy import.

    Default model: all-MiniLM-L6-v2 (384 dims, ~80MB).
    Requires `pip install sentence-transformers torch`.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers torch"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim: int = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, row)) for row in vectors]


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Factory: build the configured embedding provider."""
    s = settings or get_settings()
    if s.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedding(model_name=s.embedding_model)
    return LocalHashEmbedding(dim=s.embedding_dim)
