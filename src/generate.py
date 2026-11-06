"""Answer generation with citations and abstention.

- `MockLLM`: deterministic offline generator for tests/demos. Extracts
  the most query-relevant sentences from context and formats them with
  [1], [2] citations.
- `OpenAILLM`: production chat-completions generator with a
  citation-enforcing system prompt.

Abstention policy: if no context passes `min_context_docs`, or the best
retrieval score is below `abstain_threshold`, the assistant replies with
a fixed abstention message instead of hallucinating.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from src.config import Settings, get_settings
from src.retrieval import RetrievalResult

ABSTENTION_MESSAGE = (
    "I couldn't find relevant information in the provided documents "
    "to answer that question reliably."
)

SYSTEM_PROMPT = """You are a precise documentation assistant. Answer ONLY from the provided context passages.
Rules:
1. Every factual claim must carry a citation like [1], [2] referring to the passage numbers.
2. If the context does not contain the answer, reply exactly: "{abstain}"
3. Be concise. Use bullet points when listing items.
4. Never invent policies, dates, or numbers not present in the context.
""".format(abstain=ABSTENTION_MESSAGE)

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[int] = field(default_factory=list)
    abstained: bool = False
    model: str = "mock"
    context_used: int = 0


class BaseLLM(ABC):
    """LLM interface."""

    name: str = "base"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return a completion for the given messages."""

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Stream completion tokens; default falls back to single chunk."""
        yield self.complete(system, user)


class MockLLM(BaseLLM):
    """Deterministic offline LLM.

    Picks the top-3 sentences from context ranked by query-term overlap
    and renders them with citations. Fully deterministic — ideal for tests.
    """

    name = "mock"

    def complete(self, system: str, user: str) -> str:
        # Parse passages back out of the prompt: format "[n] passage..."
        passages: list[tuple[int, str]] = []
        for m in re.finditer(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\n\nQuestion:|\Z)", user, re.DOTALL):
            passages.append((int(m.group(1)), m.group(2).strip()))
        q_match = re.search(r"Question:\s*(.*)", user, re.DOTALL)
        query = q_match.group(1).strip() if q_match else ""
        q_terms = set(_TOKEN_RE.findall(query.lower()))

        scored: list[tuple[int, int, str]] = []  # (overlap, passage_no, sentence)
        for no, text in passages:
            for sent in _SENT_RE.split(text):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                overlap = len(q_terms & set(_TOKEN_RE.findall(sent.lower())))
                scored.append((overlap, no, sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for s in scored if s[0] > 0][:3] or scored[:2]
        if not top:
            return ABSTENTION_MESSAGE
        lines = [f"- {sent} [{no}]" for _, no, sent in top]
        return "Based on the documentation:\n" + "\n".join(lines)


class OpenAILLM(BaseLLM):
    """OpenAI chat-completions generator."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 1024, temperature: float = 0.2, base_url: str = "https://api.openai.com/v1") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("openai is not installed. Run: pip install openai") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self._client.chat.completions.stream(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) as s:
            for event in s:
                if event.type == "content.delta" and event.delta:
                    yield event.delta


def build_user_prompt(query: str, contexts: list[RetrievalResult]) -> str:
    """Render numbered context passages + question for the LLM."""
    parts = ["Context passages:"]
    for i, ctx in enumerate(contexts, start=1):
        parts.append(f"[{i}] (source: {os.path.basename(ctx.source)} | section: {ctx.section})\n{ctx.text}")
    parts.append(f"\nQuestion: {query}")
    return "\n\n".join(parts)


def extract_citations(answer: str) -> list[int]:
    """Extract sorted unique [n] citation numbers from an answer."""
    return sorted({int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)})


def should_abstain(contexts: list[RetrievalResult], settings: Settings) -> bool:
    """Decide abstention from retrieval quality signals."""
    if len(contexts) < settings.min_context_docs:
        return True
    if not contexts:
        return True
    best = max(c.score for c in contexts)
    # RRF scores are small (~0.01-0.03); rerank scores are 0-1 or 0-10.
    # Normalize: if best looks like raw RRF, scale threshold accordingly.
    threshold = settings.abstain_threshold
    if best < 0.1:
        threshold = min(threshold, 0.008)
    return best < threshold


def get_llm(settings: Settings | None = None) -> BaseLLM:
    """Factory: OpenAI when configured, else deterministic mock."""
    s = settings or get_settings()
    if s.llm_provider == "openai" and s.openai_api_key:
        try:
            return OpenAILLM(
                api_key=s.openai_api_key,
                model=s.openai_chat_model,
                max_tokens=s.generation_max_tokens,
                temperature=s.generation_temperature,
                base_url=s.openai_base_url,
            )
        except Exception:
            pass
    return MockLLM()


def generate_answer(
    query: str,
    contexts: list[RetrievalResult],
    settings: Settings | None = None,
    llm: BaseLLM | None = None,
) -> GeneratedAnswer:
    """Generate a cited answer or abstain when context is insufficient."""
    s = settings or get_settings()
    model = llm or get_llm(s)
    if should_abstain(contexts, s):
        return GeneratedAnswer(
            answer=ABSTENTION_MESSAGE, citations=[], abstained=True,
            model=model.name, context_used=0,
        )
    prompt = build_user_prompt(query, contexts)
    text = model.complete(SYSTEM_PROMPT, prompt).strip()
    if ABSTENTION_MESSAGE.lower() in text.lower():
        return GeneratedAnswer(answer=text, citations=[], abstained=True,
                               model=model.name, context_used=len(contexts))
    return GeneratedAnswer(
        answer=text,
        citations=extract_citations(text),
        abstained=False,
        model=model.name,
        context_used=len(contexts),
    )


def generate_answer_stream(
    query: str,
    contexts: list[RetrievalResult],
    settings: Settings | None = None,
    llm: BaseLLM | None = None,
) -> Iterator[str]:
    """Streaming variant yielding answer chunks (for SSE)."""
    s = settings or get_settings()
    model = llm or get_llm(s)
    if should_abstain(contexts, s):
        yield ABSTENTION_MESSAGE
        return
    prompt = build_user_prompt(query, contexts)
    yield from model.stream(SYSTEM_PROMPT, prompt)
