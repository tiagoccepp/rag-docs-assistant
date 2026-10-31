"""Reranking stage.

Provider chain with graceful fallback:
  1. Cohere Rerank API (if configured + installed)
  2. OpenAI-based pointwise rerank (if configured + installed)
  3. LocalReranker — dependency-free token-overlap + position + RRF-signal
     scorer. Always available; used in tests and offline mode.

`rerank()` wraps the chain: any provider failure falls back to local.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from src.config import Settings, get_settings
from src.retrieval import RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BaseReranker(ABC):
    """Reranker interface."""

    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        """Reorder results by relevance to query, truncated to top_k."""


class LocalReranker(BaseReranker):
    """Dependency-free reranker.

    Score = 0.6 * coverage + 0.25 * bm25ish + 0.15 * rrf_prior
      - coverage: fraction of query terms present in the passage.
      - bm25ish: saturated TF over query terms, length-normalized.
      - rrf_prior: normalized original RRF score (preserves retrieval signal).
    """

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        if not results:
            return []
        q_terms = _tokens(query)
        q_set = set(q_terms)
        max_rrf = max((r.score for r in results), default=1.0) or 1.0

        scored: list[tuple[float, RetrievalResult]] = []
        for r in results:
            doc_terms = _tokens(r.text)
            doc_counts = Counter(doc_terms)
            if not q_set or not doc_terms:
                total = 0.15 * (r.score / max_rrf)
                scored.append((total, r))
                continue
            hits = sum(1 for t in q_set if t in doc_counts)
            coverage = hits / len(q_set)
            tf_sum = sum(doc_counts.get(t, 0) for t in q_set)
            bm25ish = tf_sum / (tf_sum + 1.5 * (0.25 + 0.75 * len(doc_terms) / 200))
            prior = r.score / max_rrf
            total = 0.6 * coverage + 0.25 * bm25ish + 0.15 * prior
            scored.append((total, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [r for _, r in scored[:top_k]]
        # Refresh score to rerank score for transparency
        for (s, _), r in zip(scored[:top_k], out):
            r.score = round(float(s), 4)
        return out


class CohereReranker(BaseReranker):
    """Cohere Rerank API (e.g. rerank-english-v3.0)."""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        try:
            import cohere  # type: ignore
        except ImportError as exc:
            raise ImportError("cohere is not installed. Run: pip install cohere") from exc
        if not api_key:
            raise ValueError("COHERE_API_KEY is empty")
        self._client = cohere.Client(api_key)
        self.model = model

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        if not results:
            return []
        docs = [r.text for r in results]
        resp = self._client.rerank(model=self.model, query=query, documents=docs, top_n=min(top_k, len(docs)))
        out: list[RetrievalResult] = []
        for item in resp.results:
            r = results[item.index]
            r.score = round(float(item.relevance_score), 4)
            out.append(r)
        return out


class OpenAIReranker(BaseReranker):
    """Pointwise LLM rerank via OpenAI chat completions.

    Asks the model to score each (query, passage) pair 0-10.
    Slower than Cohere; kept as an optional fallback.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("openai is not installed. Run: pip install openai") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def _score_one(self, query: str, passage: str) -> float:
        prompt = (
            "Rate the relevance of the passage to the query on a 0-10 scale. "
            "Reply with ONLY the number.\n\n"
            f"Query: {query}\nPassage: {passage[:1500]}"
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "0").strip()
        try:
            return max(0.0, min(10.0, float(raw.split()[0])))
        except ValueError:
            return 0.0

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        scored = [(self._score_one(query, r.text), r) for r in results]
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [r for _, r in scored[:top_k]]
        for (s, _), r in zip(scored[:top_k], out):
            r.score = round(float(s), 4)
        return out


def get_reranker(settings: Settings | None = None) -> BaseReranker:
    """Build the configured reranker, falling back to local on any failure."""
    s = settings or get_settings()
    try:
        if s.rerank_provider == "cohere" and s.cohere_api_key:
            return CohereReranker(api_key=s.cohere_api_key, model=s.cohere_rerank_model)
        if s.rerank_provider == "openai" and s.openai_api_key:
            return OpenAIReranker(api_key=s.openai_api_key, model=s.openai_rerank_model)
    except Exception:
        pass
    return LocalReranker()


def rerank(
    query: str,
    results: list[RetrievalResult],
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[RetrievalResult]:
    """Rerank with automatic fallback to the local reranker on error."""
    primary = get_reranker(settings)
    try:
        return primary.rerank(query, results, top_k=top_k)
    except Exception:
        return LocalReranker().rerank(query, results, top_k=top_k)
