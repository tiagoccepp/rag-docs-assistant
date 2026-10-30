"""Hybrid retrieval: BM25 (FTS) + vector search fused with RRF.

Two backends:
  - `InMemoryRetriever`: pure-Python BM25 + cosine similarity. Used for
    tests, demos, and offline mode. No database required.
  - `PostgresRetriever`: production backend using pgvector (`<=>`) for
    vector search and `ts_rank` for full-text search, fused in Python
    with Reciprocal Rank Fusion.

RRF: score(d) = sum_over_lists( 1 / (k + rank) )
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.embeddings import EmbeddingProvider, cosine_similarity

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    source: str
    section: str
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked ID lists with RRF.

    Args:
        ranked_lists: each inner list holds doc IDs in rank order (best first).
        k: RRF smoothing constant (standard default 60).
    Returns:
        Mapping of doc ID -> fused score (higher is better).
    """
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class SimpleBM25:
    """Minimal dependency-free BM25 implementation (Robertson et al.).

    Used by InMemoryRetriever so tests run without rank-bm25 installed.
    For production scale, Postgres FTS is preferred (see PostgresRetriever).
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.n)
        # Document frequency per term
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.df = df
        # Term frequencies per doc
        self.tf: list[dict[str, int]] = []
        for doc in docs:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # BM25+ style floor to avoid negative idf on tiny corpora
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        doc_len = len(self.docs[doc_idx])
        total = 0.0
        for term in query_terms:
            tf = self.tf[doc_idx].get(term, 0)
            if tf == 0:
                continue
            idf = self.idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avgdl))
            total += idf * (tf * (self.k1 + 1)) / denom
        return total

    def rank(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return (doc_idx, score) sorted by score desc, filtered to score > 0."""
        q_terms = _tokenize(query)
        scored = [(i, self.score(q_terms, i)) for i in range(self.n)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    source: str
    section: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryRetriever:
    """Offline hybrid retriever: BM25 + cosine fused with RRF.

    The workhorse for tests and local demos. Deterministic given a
    fixed embedding provider.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.k1 = k1
        self.b = b
        self._chunks: list[StoredChunk] = []
        self._bm25: SimpleBM25 | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add_chunks(
        self,
        texts: list[str],
        sources: list[str] | None = None,
        sections: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add chunks, embedding them in one batch. Returns chunk IDs."""
        sources = sources or ["unknown"] * len(texts)
        sections = sections or [""] * len(texts)
        metadatas = metadatas or [{} for _ in texts]
        embeddings = self.embedder.embed_texts(texts)
        ids: list[str] = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            cid = f"chunk-{len(self._chunks)}"
            self._chunks.append(
                StoredChunk(
                    chunk_id=cid,
                    text=text,
                    source=sources[i],
                    section=sections[i],
                    embedding=emb,
                    metadata=metadatas[i],
                )
            )
            ids.append(cid)
        self._rebuild_bm25()
        return ids

    def _rebuild_bm25(self) -> None:
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = SimpleBM25(tokenized, k1=self.k1, b=self.b)

    def vector_search(self, query: str, top_k: int = 20) -> list[tuple[StoredChunk, float]]:
        """Cosine-similarity search over stored embeddings."""
        q = self.embedder.embed_query(query)
        scored = [(c, cosine_similarity(q, c.embedding)) for c in self._chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def bm25_search(self, query: str, top_k: int = 20) -> list[tuple[StoredChunk, float]]:
        """BM25 lexical search."""
        if self._bm25 is None:
            return []
        ranked = self._bm25.rank(query, top_k=top_k)
        return [(self._chunks[i], s) for i, s in ranked]

    def hybrid_search(
        self,
        query: str,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
    ) -> list[RetrievalResult]:
        """Hybrid search: vector + BM25 lists fused with RRF."""
        vec_hits = self.vector_search(query, top_k=top_k_vector)
        bm25_hits = self.bm25_search(query, top_k=top_k_bm25)

        vec_ids = [c.chunk_id for c, _ in vec_hits]
        bm25_ids = [c.chunk_id for c, _ in bm25_hits]
        vec_rank = {cid: r + 1 for r, cid in enumerate(vec_ids)}
        bm25_rank = {cid: r + 1 for r, cid in enumerate(bm25_ids)}

        fused = reciprocal_rank_fusion([vec_ids, bm25_ids], k=self.rrf_k)
        by_id = {c.chunk_id: c for c in self._chunks}
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k_final]
        return [
            RetrievalResult(
                chunk_id=cid,
                text=by_id[cid].text,
                source=by_id[cid].source,
                section=by_id[cid].section,
                score=score,
                vector_rank=vec_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
                metadata=by_id[cid].metadata,
            )
            for cid, score in ordered
        ]


# ---------------------------------------------------------------------------
# Postgres backend (production)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'unknown',
    section     TEXT NOT NULL DEFAULT '',
    embedding   vector(%(dim)s),
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
"""


class PostgresRetriever:
    """Production hybrid retriever on Postgres + pgvector + FTS.

    Vector search uses cosine distance (`<=>`); lexical search uses
    `ts_rank` over a generated `tsv` column. Fusion happens in Python
    with RRF so ranking policy stays in one place.

    Requires `DATABASE_URL` and the pgvector extension. Not used in tests.
    """

    def __init__(self, dsn: str, embedder: EmbeddingProvider, dim: int = 384, rrf_k: int = 60) -> None:
        self.dsn = dsn
        self.embedder = embedder
        self.dim = dim
        self.rrf_k = rrf_k

    def _connect(self):  # lazy import so tests don't need psycopg2
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore

        conn = psycopg2.connect(self.dsn)
        conn.autocommit = True
        return conn

    def init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL, {"dim": self.dim})
        finally:
            conn.close()

    def add_chunks(
        self,
        texts: list[str],
        sources: list[str] | None = None,
        sections: list[str] | None = None,
    ) -> list[str]:
        import json
        import uuid

        sources = sources or ["unknown"] * len(texts)
        sections = sections or [""] * len(texts)
        embeddings = self.embedder.embed_texts(texts)
        ids = [f"chunk-{uuid.uuid4().hex[:12]}" for _ in texts]
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                for cid, text, emb, src, sec in zip(ids, texts, embeddings, sources, sections):
                    cur.execute(
                        """INSERT INTO chunks (chunk_id, text, source, section, embedding, metadata)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ON CONFLICT (chunk_id) DO UPDATE
                           SET text=EXCLUDED.text, embedding=EXCLUDED.embedding""",
                        (cid, text, src, sec, emb, json.dumps({})),
                    )
        finally:
            conn.close()
        return ids

    def hybrid_search(
        self,
        query: str,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
    ) -> list[RetrievalResult]:
        """Run vector + FTS queries and fuse with RRF."""
        import psycopg2.extras  # type: ignore

        q_emb = self.embedder.embed_query(query)
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """SELECT chunk_id, text, source, section, metadata,
                              embedding <=> %s::vector AS distance
                       FROM chunks ORDER BY distance ASC LIMIT %s""",
                    (q_emb, top_k_vector),
                )
                vec_rows = list(cur.fetchall())
                cur.execute(
                    """SELECT chunk_id, text, source, section, metadata,
                              ts_rank(tsv, plainto_tsquery('english', %s)) AS rank
                       FROM chunks
                       WHERE tsv @@ plainto_tsquery('english', %s)
                       ORDER BY rank DESC LIMIT %s""",
                    (query, query, top_k_bm25),
                )
                fts_rows = list(cur.fetchall())
        finally:
            conn.close()

        vec_ids = [r["chunk_id"] for r in vec_rows]
        fts_ids = [r["chunk_id"] for r in fts_rows]
        vec_rank = {cid: r + 1 for r, cid in enumerate(vec_ids)}
        fts_rank = {cid: r + 1 for r, cid in enumerate(fts_ids)}
        fused = reciprocal_rank_fusion([vec_ids, fts_ids], k=self.rrf_k)
        by_id = {r["chunk_id"]: r for r in (vec_rows + fts_rows)}
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k_final]
        return [
            RetrievalResult(
                chunk_id=cid,
                text=by_id[cid]["text"],
                source=by_id[cid]["source"],
                section=by_id[cid].get("section", ""),
                score=score,
                vector_rank=vec_rank.get(cid),
                bm25_rank=fts_rank.get(cid),
                metadata=dict(by_id[cid].get("metadata", {}) or {}),
            )
            for cid, score in ordered
        ]
