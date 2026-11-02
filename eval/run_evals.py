"""Run RAGAS-style offline evals over the golden QA set.

Usage:
    python eval/run_evals.py [--min-mean 0.35] [--output eval/report.json]

Pipeline per question: hybrid retrieve -> local rerank -> mock generate,
then score faithfulness / relevancy / context precision / recall.
Fully offline and deterministic. Exits non-zero if mean < --min-mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.embeddings import LocalHashEmbedding
from src.evals import aggregate, evaluate_case
from src.generate import MockLLM, build_user_prompt
from src.ingest import ingest_directory
from src.rerank import LocalReranker
from src.retrieval import InMemoryRetriever


def load_golden(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) >= 10, "golden set too small"
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline RAG evals")
    parser.add_argument("--min-mean", type=float, default=0.35, help="CI gate threshold")
    parser.add_argument("--output", type=str, default="eval/report.json", help="Report path")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = Settings()
    embedder = LocalHashEmbedding(dim=settings.embedding_dim)
    retriever = InMemoryRetriever(embedder=embedder, k1=settings.bm25_k1, b=settings.bm25_b, rrf_k=settings.rrf_k)
    reranker = LocalReranker()
    llm = MockLLM()

    chunks = ingest_directory(ROOT / "data" / "sample_docs",
                              chunk_size=settings.chunk_size_tokens,
                              overlap=settings.chunk_overlap_tokens)
    retriever.add_chunks(
        [c.text for c in chunks],
        sources=[c.source for c in chunks],
        sections=[c.section for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks from sample docs.")

    golden = load_golden(ROOT / "eval" / "golden_qa.json")
    results = []
    for item in golden:
        q = item["question"]
        keywords = item.get("expected_keywords", [])
        hits = retriever.hybrid_search(q, top_k_vector=10, top_k_bm25=10, top_k_final=10)
        reranked = reranker.rerank(q, hits, top_k=args.top_k)
        ctxs = [r.text for r in reranked]
        prompt = build_user_prompt(q, reranked)
        answer = llm.complete("", prompt)
        res = evaluate_case(q, answer, ctxs, keywords, abstained=False)
        results.append(res)
        print(f"  [{res.mean:.2f}] {q[:70]}")

    agg = aggregate(results)
    print("\nAggregate:", json.dumps(agg, indent=2))

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg,
                   "cases": [{"question": r.question, "faithfulness": r.faithfulness,
                              "relevancy": r.relevancy, "context_precision": r.context_precision,
                              "context_recall": r.context_recall, "mean": r.mean}
                             for r in results]}, f, indent=2)
    print(f"Report written to {out_path}")

    if agg["mean"] < args.min_mean:
        print(f"FAIL: mean {agg['mean']:.3f} < gate {args.min_mean}")
        return 1
    print(f"PASS: mean {agg['mean']:.3f} >= gate {args.min_mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
