# Live verification

Verified against a real LLM API via `scripts/live_smoke.py` (exactly one real
call, stdlib only, key never printed).

- Date: 2026-11-05
- Provider: opencode Zen — OpenAI-compatible `https://opencode.ai/zen/go/v1`
- Model: `beam-free` (free tier, $0)
- Key: local file, shown redacted below

```text
provider=https://opencode.ai/zen/go/v1 model=beam-free key=*** latency_ms=4058
cited=True
--- answer ---
The core collaboration hours are **10:00 to 15:00 America/Montevideo**, Monday to Friday [1].
LIVE_SMOKE=OK
```

What this proves: the generation path works end-to-end against a real model,
and the model honored the cite-or-abstain contract (the `[1]` citation is
present and the fact matches the source chunk).
