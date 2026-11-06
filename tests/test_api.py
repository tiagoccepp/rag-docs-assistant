"""API tests (FastAPI TestClient). Fully offline via in-memory backend."""

import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.config import Settings
from src.embeddings import LocalHashEmbedding
from src.retrieval import InMemoryRetriever


@pytest.fixture()
def client(monkeypatch):
    """TestClient with a fresh seeded in-memory retriever."""
    settings = Settings(embedding_dim=64)
    retriever = InMemoryRetriever(embedder=LocalHashEmbedding(dim=64))
    retriever.add_chunks(
        texts=[
            "Employees receive 20 vacation days per year with monthly accrual.",
            "The Team plan costs USD 12 per user per month.",
        ],
        sources=["handbook.md", "faq.md"],
        sections=["Time Off", "Pricing"],
    )
    monkeypatch.setattr(api_module, "_retriever", retriever)
    monkeypatch.setattr(api_module, "_seed_sample_docs", lambda: 0)
    with TestClient(api_module.app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_stats_reports_backend(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json()["backend"] == "InMemoryRetriever"
    assert resp.json()["chunks"] == 2


def test_query_returns_cited_answer(client):
    resp = client.post("/query", json={"question": "How many vacation days?", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert "20" in body["answer"]
    assert body["abstained"] is False
    assert len(body["passages"]) >= 1
    assert body["passages"][0]["source"] == "handbook.md"


def test_query_validates_short_question(client):
    resp = client.post("/query", json={"question": "ab"})
    assert resp.status_code == 422


def test_ingest_adds_chunks(client):
    md = "## Travel\n\nHotel cap is USD 180 per night for all employees."
    resp = client.post("/ingest", json={"markdown": md, "source": "policy.md"})
    assert resp.status_code == 200
    assert resp.json()["chunks_added"] >= 1
    # New content is retrievable
    q = client.post("/query", json={"question": "What is the hotel cap?", "top_k": 3})
    assert "180" in q.json()["answer"]


def test_ingest_rejects_empty(client):
    resp = client.post("/ingest", json={"markdown": "   ", "source": "empty.md"})
    assert resp.status_code == 400


def test_ingest_disabled_on_public_demos(client, monkeypatch):
    monkeypatch.setenv("DISABLE_INGEST", "1")
    resp = client.post("/ingest", json={"markdown": "## X\n\nY", "source": "evil.md"})
    assert resp.status_code == 403


def test_question_max_length(client):
    resp = client.post("/query", json={"question": "x" * 2001})
    assert resp.status_code == 422


def test_query_stream_emits_events(client):
    resp = client.post("/query/stream", json={"question": "Team plan cost?", "top_k": 2})
    assert resp.status_code == 200
    text = resp.text
    assert "passages" in text
    assert "done" in text
