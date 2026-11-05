# Database schema

Default backend is in-memory (zero setup). Set `USE_POSTGRES=1` + `DATABASE_URL`
for the Postgres + pgvector backend, created by `sql/init.sql`:

```sql
CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'unknown',
    section     TEXT NOT NULL DEFAULT '',
    embedding   vector(384),
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);
```

- `embedding` + HNSW index serve vector search; `tsv` + GIN serve BM25 full-text.
- `metadata` JSONB is reserved for author/date/sensitivity labels (filtering is future work).
- `source` + `section` power citations: every generated claim links back to a chunk.
