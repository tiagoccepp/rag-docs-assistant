# Security & confidentiality notes

Honest accounting of what this PoC does and does not do with sensitive data.

## Done

- Secrets (API keys) come from env/`.env` only; `.env` is gitignored and the
  live smoke script prints only the key's last 4 chars.
- The `metadata` JSONB column is reserved for sensitivity labels so a future
  clearance filter can exclude restricted docs per caller.
- Abstention gate: the system says "I don't know" instead of guessing, which
  also bounds leakage from low-relevance passages.

## Not done (future work)

- No PII redaction on ingest or retrieval — documents are indexed as-is.
  Production use with customer docs needs redaction + per-caller clearance
  filtering before this is safe.
- No audit log of who asked what; add query logging with retention policy.
- Sample docs are synthetic (`data/sample_docs/`); no real customer data anywhere.
