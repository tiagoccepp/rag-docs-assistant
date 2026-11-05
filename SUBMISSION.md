# RAG Docs Assistant — submission one-pager

Production-grade **Retrieval-Augmented Generation** assistant over company docs: hybrid search (BM25 + vector fused with RRF), cross-encoder-style rerank, cited answers, abstention when docs lack the answer, RAGAS-style evals gating deploys.

## Architecture (5 lines)

1. Ingest: Markdown docs → structure-aware 512-token chunks → embeddings (local or MiniLM).
2. Retrieve: BM25 full-text + vector cosine, fused with Reciprocal Rank Fusion (top-20 each).
3. Rerank: local reranker (Cohere/OpenAI pluggable) down to top-5.
4. Generate: cited answer with `[n]` markers, or abstain ("couldn't find relevant information").
5. Serve: FastAPI with `/query` + SSE `/query/stream`; Postgres+pgvector backend via `USE_POSTGRES=1`.

## Run locally (no API keys)

```bash
./demo.sh            # boots offline API on :8000 (mock LLM, auto-seeds sample docs)
pytest -q            # 38 tests, offline
python eval/run_evals.py   # golden QA (n=32), CI gate mean >= 0.35
```

## Key files to read

- `src/retrieval.py` — hybrid BM25+vector+RRF (in-memory + Postgres)
- `src/generate.py` — citations + abstention gate
- `src/ingest.py` — structure-aware chunking
- `src/evals.py`, `eval/golden_qa.json` — faithfulness/relevancy/context metrics
- `demo/transcript.md` — verified live transcript

## Sample Q&A (from live transcript)

- "What are the core collaboration hours?" → "10:00 to 15:00 America/Montevideo…" [1], score 0.88
- "Zqxw kvetch blorpt…?" → abstained, zero citations (refused to guess)
