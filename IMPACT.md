# Why this matters to an organization

## Who feels the pain

Support teams, operations, and new hires waste hours digging through handbooks,
FAQs, and policy docs. Keyword search misses paraphrases; asking an LLM directly
gets fluent, confident, wrong answers.

## What changes with this system

- **Cited answers or abstention.** Every claim links to a source chunk, or the
  system says it doesn't know. Trust becomes verifiable instead of vibes-based.
- **Eval-gated knowledge base.** The golden QA set + faithfulness gate mean a doc
  update that degrades answers blocks deployment — the KB can't silently rot.
- **Hybrid retrieval.** Exact terms (prices, SLAs, error codes) via BM25 plus
  paraphrases via vectors: fewer "I couldn't find it" dead ends.

## Metrics a team would track

- Time-to-answer for internal questions; support ticket deflection rate.
- Faithfulness / answer-relevancy on the golden set over time.
- Abstention rate (too high = KB gaps to fill; too low = gate may be loose).
