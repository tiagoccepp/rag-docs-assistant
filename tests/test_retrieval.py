"""Tests for hybrid retrieval, rerank, generation, and evals. Fully offline."""

import pytest

from src.config import Settings
from src.embeddings import LocalHashEmbedding, cosine_similarity
from src.evals import (
    aggregate,
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_case,
    faithfulness,
)
from src.generate import (
    ABSTENTION_MESSAGE,
    GeneratedAnswer,
    MockLLM,
    build_user_prompt,
    extract_citations,
    generate_answer,
    should_abstain,
)
from src.ingest import chunk_document
from src.rerank import LocalReranker, rerank
from src.retrieval import (
    InMemoryRetriever,
    RetrievalResult,
    reciprocal_rank_fusion,
)


@pytest.fixture()
def settings():
    return Settings(
        embedding_dim=64,
        chunk_size_tokens=100,
        chunk_overlap_tokens=10,
        abstain_threshold=0.35,
        min_context_docs=1,
    )


@pytest.fixture()
def retriever(settings):
    r = InMemoryRetriever(embedder=LocalHashEmbedding(dim=64))
    r.add_chunks(
        texts=[
            "The vacation policy grants 20 vacation days per year for full-time employees.",
            "The Team plan costs USD 12 per user per month with unlimited retention.",
            "On-call rotates weekly from Monday 09:00 UTC in PagerDuty.",
        ],
        sources=["handbook.md", "faq.md", "oncall.md"],
        sections=["Time Off", "Pricing", "Rotation"],
    )
    return r


# --- embeddings ---

def test_local_hash_embedding_deterministic():
    emb = LocalHashEmbedding(dim=32)
    a = emb.embed_texts(["hello world"])[0]
    b = emb.embed_texts(["hello world"])[0]
    assert a == b
    assert len(a) == 32


def test_cosine_similarity_identical_is_one():
    emb = LocalHashEmbedding(dim=32)
    v = emb.embed_query("some query text")
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_dim_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


# --- RRF ---

def test_rrf_prefers_top_ranked_everywhere():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]], k=60)
    # 'a' ranks 1+2, 'b' ranks 2+1 -> tie; both beat 'c' (3+3)
    assert fused["a"] == pytest.approx(fused["b"])
    assert fused["a"] > fused["c"]


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == {}


# --- retriever ---

def test_hybrid_search_finds_relevant_chunk(retriever):
    results = retriever.hybrid_search("How many vacation days?", top_k_final=1)
    assert len(results) == 1
    assert results[0].source == "handbook.md"
    assert "vacation" in results[0].text.lower()


def test_hybrid_search_returns_ranks(retriever):
    results = retriever.hybrid_search("Team plan pricing", top_k_final=3)
    assert len(results) >= 1
    assert all(r.score > 0 for r in results)


def test_vector_search_orders_by_similarity(retriever):
    hits = retriever.vector_search("PagerDuty on-call rotation schedule", top_k=3)
    assert hits[0][0].source == "oncall.md"


def test_bm25_search_matches_keywords(retriever):
    hits = retriever.bm25_search("USD 12 per user per month", top_k=3)
    assert hits
    assert hits[0][0].source == "faq.md"


# --- rerank ---

def test_local_reranker_promotes_query_match(retriever):
    results = retriever.hybrid_search("vacation days policy", top_k_final=3)
    reranked = LocalReranker().rerank("vacation days policy", results, top_k=3)
    assert reranked[0].source == "handbook.md"


def test_rerank_fallback_chain_returns_results(retriever, settings):
    results = retriever.hybrid_search("on-call", top_k_final=3)
    out = rerank("on-call", results, top_k=2, settings=settings)
    assert len(out) == 2


def test_rerank_empty_input():
    assert LocalReranker().rerank("q", [], top_k=5) == []


# --- generation ---

def test_generate_answer_includes_citations(retriever, settings):
    results = retriever.hybrid_search("vacation days", top_k_final=3)
    ans = generate_answer("How many vacation days?", results, settings=settings, llm=MockLLM())
    assert isinstance(ans, GeneratedAnswer)
    assert not ans.abstained
    assert ans.citations, "expected [n] citations in answer"
    assert "20" in ans.answer


def test_generate_answer_abstains_without_context(settings):
    ans = generate_answer("Anything?", [], settings=settings, llm=MockLLM())
    assert ans.abstained
    assert ans.answer == ABSTENTION_MESSAGE


def test_should_abstain_on_empty(settings):
    assert should_abstain([], settings) is True


def test_extract_citations():
    assert extract_citations("Claim one [1] and two [2], again [1].") == [1, 2]
    assert extract_citations("no citations") == []


def test_build_user_prompt_numbers_passages(retriever):
    results = retriever.hybrid_search("vacation", top_k_final=2)
    prompt = build_user_prompt("Q?", results)
    assert "[1]" in prompt and "Question: Q?" in prompt


# --- evals ---

def test_faithfulness_supported_claims():
    ctx = ["Employees receive 20 vacation days per year with monthly accrual."]
    ans = "Employees receive 20 vacation days per year."
    assert faithfulness(ans, ctx) == pytest.approx(1.0)


def test_faithfulness_unsupported_is_zero():
    assert faithfulness("The moon is made of cheese.", ["Vacation policy details here."]) == 0.0


def test_answer_relevancy_covers_question():
    assert answer_relevancy("Vacation days are 20 per year.", "How many vacation days?") >= 0.5


def test_context_precision_and_recall():
    ctxs = ["20 vacation days per year", "unrelated pricing text here"]
    assert context_precision(ctxs, ["vacation", "20"], k=2) == pytest.approx(0.5)
    assert context_recall(ctxs, ["vacation", "20"]) == pytest.approx(1.0)


def test_evaluate_case_and_aggregate():
    r = evaluate_case("q", "answer text here", ["answer text here"], ["answer"])
    assert 0.0 <= r.mean <= 1.0
    agg = aggregate([r, r])
    assert agg["n"] == 2
    assert agg["mean"] == pytest.approx(r.mean)


def test_chunk_document_integration_small():
    chunks = chunk_document("## A\n\nAlpha beta gamma delta.", source="x.md",
                            chunk_size=10, overlap=2)
    assert len(chunks) >= 1
    assert chunks[0].section == "A"
