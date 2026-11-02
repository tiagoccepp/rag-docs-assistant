"""RAGAS-style offline evaluation metrics.

Implements lightweight, dependency-free versions of:
  - Faithfulness: fraction of answer claims supported by retrieved context.
  - Answer relevancy: query-term coverage of the answer.
  - Context precision@k: fraction of retrieved passages relevant to the
    expected answer (via expected keyword hits).
  - Context recall: fraction of expected keywords covered by any passage.

"Claims" are approximated as sentences; "relevance" via normalized token
overlap. Deterministic and offline — suitable for CI gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "it", "this", "that", "you", "your",
    "we", "our", "as", "by", "at", "from", "can", "all", "any",
})


def content_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens minus stopwords."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def split_claims(answer: str) -> list[str]:
    """Split an answer into claim-like sentences."""
    parts = [s.strip(" -\n\t") for s in _SENT_RE.split(answer.strip())]
    return [p for p in parts if len(content_tokens(p)) >= 2]


def _overlap_ratio(claim_tokens: set[str], context_tokens: set[str]) -> float:
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & context_tokens) / len(claim_tokens)


def faithfulness(answer: str, contexts: list[str], threshold: float = 0.5) -> float:
    """Fraction of answer claims supported by the union of contexts.

    A claim is supported if >= `threshold` of its content tokens appear
    in the combined context.
    """
    claims = split_claims(answer)
    if not claims:
        return 0.0
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens |= content_tokens(c)
    supported = sum(1 for cl in claims if _overlap_ratio(content_tokens(cl), ctx_tokens) >= threshold)
    return supported / len(claims)


def answer_relevancy(answer: str, question: str) -> float:
    """Fraction of question content tokens reflected in the answer."""
    q = content_tokens(question)
    if not q:
        return 0.0
    a: set[str] = set()
    for tok in _TOKEN_RE.findall(answer.lower()):
        a.add(tok)
    return len(q & a) / len(q)


def context_precision(contexts: list[str], expected_keywords: list[str], k: int = 5) -> float:
    """Fraction of top-k passages hitting at least one expected keyword."""
    if not contexts or not expected_keywords:
        return 0.0
    keys = {kw.lower() for kw in expected_keywords}
    top = contexts[:k]
    hits = 0
    for ctx in top:
        toks = set(_TOKEN_RE.findall(ctx.lower()))
        if toks & keys:
            hits += 1
    return hits / len(top)


def context_recall(contexts: list[str], expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in any retrieved passage."""
    if not expected_keywords:
        return 0.0
    keys = {kw.lower() for kw in expected_keywords}
    all_toks: set[str] = set()
    for ctx in contexts:
        all_toks |= set(_TOKEN_RE.findall(ctx.lower()))
    return len(keys & all_toks) / len(keys)


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]
    expected_answer: str = ""


@dataclass
class EvalResult:
    question: str
    faithfulness: float
    relevancy: float
    context_precision: float
    context_recall: float
    abstained: bool = False
    details: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return (self.faithfulness + self.relevancy + self.context_precision + self.context_recall) / 4


def evaluate_case(
    question: str,
    answer: str,
    contexts: list[str],
    expected_keywords: list[str],
    abstained: bool = False,
) -> EvalResult:
    """Score a single QA case on all four metrics."""
    return EvalResult(
        question=question,
        faithfulness=0.0 if abstained else faithfulness(answer, contexts),
        relevancy=0.0 if abstained else answer_relevancy(answer, question),
        context_precision=context_precision(contexts, expected_keywords),
        context_recall=context_recall(contexts, expected_keywords),
        abstained=abstained,
    )


def aggregate(results: list[EvalResult]) -> dict[str, float]:
    """Mean-aggregate a list of EvalResults."""
    if not results:
        return {"faithfulness": 0.0, "relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0, "mean": 0.0, "n": 0}
    n = len(results)
    return {
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "relevancy": sum(r.relevancy for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "context_recall": sum(r.context_recall for r in results) / n,
        "mean": sum(r.mean for r in results) / n,
        "n": n,
    }
