"""FastAPI application: ingest, query, and SSE streaming query.

Run:
    uvicorn src.api:app --reload --port 8000

Modes:
  - Offline (default): InMemoryRetriever + MockLLM. Zero external deps.
  - Production: set USE_POSTGRES=1 and DATABASE_URL for pgvector backend,
    LLM_PROVIDER=openai + OPENAI_API_KEY for real generation.

On startup in offline mode the API auto-ingests `data/sample_docs/*.md`
so the demo works immediately.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import Settings, get_settings
from src.embeddings import get_embedding_provider
from src.generate import generate_answer, generate_answer_stream
from src.ingest import chunk_document, ingest_directory
from src.ratelimit import RateLimitMiddleware
from src.rerank import rerank
from src.retrieval import InMemoryRetriever, PostgresRetriever, RetrievalResult

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Docs Assistant",
    description="Hybrid-search RAG over company docs: pgvector + FTS + RRF, rerank, cited answers.",
    version="0.1.0",
)
app.add_middleware(RateLimitMiddleware, prefixes=("/query", "/ingest"))

_settings = get_settings()
_retriever: InMemoryRetriever | PostgresRetriever | None = None


def get_retriever() -> InMemoryRetriever | PostgresRetriever:
    """Build (once) and return the configured retriever backend."""
    global _retriever
    if _retriever is not None:
        return _retriever
    s = get_settings()
    embedder = get_embedding_provider(s)
    if os.getenv("USE_POSTGRES", "0") == "1":
        r = PostgresRetriever(dsn=s.database_url, embedder=embedder, dim=s.embedding_dim, rrf_k=s.rrf_k)
        r.init_schema()
        _retriever = r
    else:
        _retriever = InMemoryRetriever(embedder=embedder, k1=s.bm25_k1, b=s.bm25_b, rrf_k=s.rrf_k)
    return _retriever


def _seed_sample_docs() -> int:
    """Auto-ingest bundled sample docs into an in-memory retriever."""
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    if not sample_dir.exists():
        return 0
    s = get_settings()
    chunks = ingest_directory(sample_dir, chunk_size=s.chunk_size_tokens, overlap=s.chunk_overlap_tokens)
    if not chunks:
        return 0
    r = get_retriever()
    if isinstance(r, InMemoryRetriever) and r.size > 0:
        return 0  # already seeded
    texts = [c.text for c in chunks]
    sources = [c.source for c in chunks]
    sections = [c.section for c in chunks]
    if isinstance(r, InMemoryRetriever):
        r.add_chunks(texts, sources=sources, sections=sections)
    else:
        assert isinstance(r, PostgresRetriever)
        r.add_chunks(texts, sources=sources, sections=sections)
    return len(chunks)


@app.on_event("startup")
def _on_startup() -> None:
    try:
        n = _seed_sample_docs()
        print(f"[startup] seeded {n} chunks")
    except Exception as exc:  # never crash startup on seeding
        print(f"[startup] seed skipped: {exc}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    markdown: str = Field(max_length=200000, description="Markdown document content")
    source: str = Field(default="api-upload", max_length=200, description="Source label for citations")


class IngestResponse(BaseModel):
    chunks_added: int
    source: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Final passages returned")


class PassageOut(BaseModel):
    chunk_id: str
    text: str
    source: str
    section: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    abstained: bool
    model: str
    passages: list[PassageOut]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(question: str, top_k: int, settings: Settings | None = None) -> tuple[str, list[int], bool, str, list[RetrievalResult]]:
    """Full RAG pipeline: retrieve -> rerank -> generate. Returns answer parts."""
    s = settings or get_settings()
    retriever = get_retriever()
    results = retriever.hybrid_search(
        question,
        top_k_vector=s.top_k_vector,
        top_k_bm25=s.top_k_bm25,
        top_k_final=s.top_k_rerank,
    )
    reranked = rerank(question, results, top_k=top_k, settings=s)
    gen = generate_answer(question, reranked, settings=s)
    return gen.answer, gen.citations, gen.abstained, gen.model, reranked


def _to_passages(results: list[RetrievalResult]) -> list[PassageOut]:
    return [
        PassageOut(chunk_id=r.chunk_id, text=r.text, source=r.source, section=r.section, score=round(r.score, 4))
        for r in results
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    s = get_settings()
    size = 0
    try:
        r = get_retriever()
        if isinstance(r, InMemoryRetriever):
            size = r.size
    except Exception:
        pass
    return {"status": "ok", "chunks": size, "llm": s.llm_provider, "rerank": s.rerank_provider}


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    """Minimal demo UI (single static file, no build step)."""
    return FileResponse(Path(__file__).resolve().parent / "static" / "ui.html")


@app.get("/stats")
def stats() -> dict:
    r = get_retriever()
    size = r.size if isinstance(r, InMemoryRetriever) else -1
    return {"backend": type(r).__name__, "chunks": size}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if os.getenv("DISABLE_INGEST") == "1":
        raise HTTPException(status_code=403, detail="ingest disabled on public demos")
    s = get_settings()
    chunks = chunk_document(req.markdown, source=req.source, chunk_size=s.chunk_size_tokens, overlap=s.chunk_overlap_tokens)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest")
    r = get_retriever()
    texts = [c.text for c in chunks]
    sources = [c.source for c in chunks]
    sections = [c.section for c in chunks]
    r.add_chunks(texts, sources=sources, sections=sections)  # type: ignore[arg-type]
    return IngestResponse(chunks_added=len(chunks), source=req.source)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    answer, citations, abstained, model, passages = run_pipeline(req.question, req.top_k)
    return QueryResponse(
        answer=answer, citations=citations, abstained=abstained,
        model=model, passages=_to_passages(passages),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> EventSourceResponse:
    """SSE streaming: `passages` event first, then `token` events, then `done`."""
    s = get_settings()
    retriever = get_retriever()
    results = retriever.hybrid_search(
        req.question, top_k_vector=s.top_k_vector,
        top_k_bm25=s.top_k_bm25, top_k_final=s.top_k_rerank,
    )
    reranked = rerank(req.question, results, top_k=req.top_k, settings=s)

    async def event_gen() -> AsyncIterator[dict]:
        yield {
            "event": "passages",
            "data": json.dumps([p.model_dump() for p in _to_passages(reranked)]),
        }
        loop = asyncio.get_running_loop()

        def _collect() -> list[str]:
            return list(generate_answer_stream(req.question, reranked, settings=s))

        tokens = await loop.run_in_executor(None, _collect)
        for tok in tokens:
            yield {"event": "token", "data": json.dumps({"text": tok})}
        yield {"event": "done", "data": json.dumps({"abstained": False})}

    return EventSourceResponse(event_gen())
