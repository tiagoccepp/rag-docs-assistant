# Architecture

## Goals

1. **Correctness first**: cited answers or explicit abstention — never hallucinated policy.
2. **Hybrid recall**: exact terms (prices, limits) via lexical search, paraphrases via vectors.
3. **Offline-capable core**: tests, evals, and demo run with zero network or API keys.
4. **Portfolio clarity**: raw, explicit functions over framework magic (no LlamaIndex).

## Pipeline

### 1. Ingestion (`src/ingest.py`)

- Markdown is split on `##`/`###` headings to preserve section boundaries.
- Each section is windowed into ~512-token chunks with 50-token overlap so
  sentences spanning boundaries stay recoverable.
- Every chunk carries `{source, section, position, token_count}` for precise citations.
- Token counts use `tiktoken/cl100k_base` when installed, else a 1.3× word heuristic.

Why structure-aware: naive fixed windows slice tables and procedures in half;
section headers prepended to each chunk also improve retrieval ("Time Off" context
travels with the chunk).

### 2. Embeddings (`src/embeddings.py`)

| Provider | Use | Deps |
|---|---|---|
| `LocalHashEmbedding` | Tests, CI, offline demo | None (stdlib) |
| `SentenceTransformerEmbedding` | Production quality | `sentence-transformers` + `torch` |

The local provider is signed feature hashing into 384 dims, L2-normalized —
deterministic and dependency-free. Swapping providers is one env var
(`EMBEDDING_PROVIDER`); the retriever only sees `embed_texts()`.

### 3. Retrieval (`src/retrieval.py`)

**In-memory backend** (tests/demo): pure-Python BM25 + cosine similarity.
**Postgres backend** (production): pgvector HNSW (`<=>` cosine) + `tsvector` FTS
with `ts_rank`, fused in Python.

Fusion is **Reciprocal Rank Fusion**: `score(d) = Σ 1/(k + rank)` with `k=60`.
RRF needs no score calibration between heterogeneous lists, which is exactly the
vector-vs-BM25 situation. Ranks from each list are kept on the result objects for
observability (`vector_rank`, `bm25_rank`).

Defaults: top-20 per channel → RRF → top-10 to rerank → top-5 to generate.

### 4. Rerank (`src/rerank.py`)

Chain with graceful degradation:

1. **Cohere Rerank** (`rerank-english-v3.0`) when `RERANK_PROVIDER=cohere` + key.
2. **OpenAI pointwise** scoring when `openai` + key.
3. **Local scorer** (always available): `0.6·coverage + 0.25·TF-saturation + 0.15·RRF-prior`.

Any exception in 1/2 falls back to 3. The local scorer is deliberately simple and
transparent — good enough to fix ordering mistakes from RRF on small corpora.

### 5. Generation (`src/generate.py`)

- **Abstention gate** before any LLM call: fewer than `MIN_CONTEXT_DOCS`
  passages, or best score under `ABSTAIN_THRESHOLD` → fixed abstention message.
  (Threshold auto-scales for raw RRF score magnitudes.)
- **System prompt** enforces `[n]` citations per claim and verbatim abstention
  when context is insufficient.
- **MockLLM** (offline): ranks context sentences by query-term overlap and
  renders top-3 with citations — deterministic for tests/evals.
- **OpenAILLM**: chat completions with both `complete()` and streaming `stream()`.
- `extract_citations()` parses `[n]` markers so the API returns structured citations.

### 6. API (`src/api.py`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status + chunk count + provider names |
| `GET /stats` | Backend type + index size |
| `POST /ingest` | Chunk + index one Markdown doc |
| `POST /query` | Full pipeline, JSON answer + passages |
| `POST /query/stream` | SSE: `passages` event, then `token` events, then `done` |

Backend selection: `USE_POSTGRES=1` → pgvector, else in-memory (auto-seeds
`data/sample_docs/` on startup for instant demo).

## Evaluation (`src/evals.py`, `eval/`)

RAGAS-style metrics reimplemented offline (token-overlap based, deterministic):

- **Faithfulness**: fraction of answer sentences supported (≥50% content-token
  overlap) by retrieved context.
- **Answer relevancy**: fraction of question content tokens present in the answer.
- **Context precision@k**: fraction of top-k passages hitting an expected keyword.
- **Context recall**: fraction of expected keywords found in any passage.

`eval/run_evals.py` runs 32 golden QAs end-to-end and gates on mean score
(`--min-mean`, default 0.35) for CI.

## Trade-offs & limitations

- Token-overlap metrics are proxies, not LLM-judge RAGAS — cheaper and
  deterministic, but they under-reward good paraphrases.
- Local hash embeddings capture lexical, not semantic, similarity — fine for
  tests, not for production (use MiniLM or better).
- Single-collection index; multi-tenancy would need a `workspace_id` column +
  filtered search.
- No query rewriting / HyDE — a natural next step for vague questions.

## Scaling notes

- pgvector HNSW handles ~1M chunks comfortably on one node; beyond that,
  partition by workspace or move to a dedicated vector DB.
- Rerank is the latency bottleneck (cross-encoder/API per candidate) — keep
  `TOP_K_RERANK ≤ 10` and consider batching.
- Streaming keeps time-to-first-token low even with large generations.
