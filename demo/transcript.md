# Demo transcript — rag-docs-assistant (live local server)

Server: `.venv/bin/python -m uvicorn src.api:app --port 8010` (offline: InMemoryRetriever + mock LLM, keyless).
All pairs below are verbatim from the running API (2026-11-05). Passage lists trimmed to top hit + score; full lists in API responses.

## Health + stats

GET /health:

```json
{"status":"ok","chunks":26,"llm":"mock","rerank":"local"}
```

GET /stats:

```json
{"backend":"InMemoryRetriever","chunks":26}
```

## Q1 — answerable (handbook)

Request POST /query:

```json
{"question":"What are the core collaboration hours?"}
```

Response (abridged):

```json
{
  "answer": "Based on the documentation:\n- ... Our core collaboration hours are **10:00 to 15:00 America/Montevideo**, Monday to Friday. [1]\n- Outside core hours you may arrange a flexible schedule with your manager. [1]\n- The standard work week is 40 hours. [1]",
  "citations": [1],
  "abstained": false,
  "model": "mock",
  "top_passage": {"chunk_id": "chunk-1", "section": "Working Hours", "score": 0.8847}
}
```

## Q2 — answerable (pricing)

Request: `{"question":"How much does the Team plan cost per user?"}`

Response: answered with citations [1,2,3], abstained false. Top passage: Pricing section, score 0.5723 — contains "Team: USD 12 per user per month".

## Q3 — answerable (travel policy)

Request: `{"question":"What is the hotel cap per night?"}`

Response: answered with citations [1,2,3], abstained false. Top passage: Travel section, score 0.8071 — contains "Hotel cap is USD 180 per night".

## Q4 — unanswerable (abstention gate)

Request: `{"question":"Zqxw kvetch blorpt flarnib snizzlewick?"}`

Response (verbatim):

```json
{
  "answer": "I couldn't find relevant information in the provided documents to answer that question reliably.",
  "citations": [],
  "abstained": true,
  "model": "mock"
}
```

The system abstains instead of guessing. Gate also covered by unit tests (`test_generate_answer_abstains_without_context`, `test_should_abstain_on_empty`).

## Streaming

POST /query/stream emits SSE: `event: passages` (ranked chunks) → `event: token` (answer text) → `event: done` (`{"abstained": false}`). Verified live.

## Evals (golden set, n=32)

```
context_recall: 0.755, mean: 0.577 — PASS: mean 0.577 >= gate 0.35
```

## Tests

```
38 passed, 3 warnings
```
