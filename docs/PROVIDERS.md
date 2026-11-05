# LLM providers

The generator speaks **OpenAI Chat Completions**, so any OpenAI-compatible
endpoint works. Tests, evals and the demo run offline (`LLM_PROVIDER=mock`);
set `LLM_PROVIDER=openai` + key to use a real model.

## Supported endpoints

| Provider | Base URL | Key | Cost |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` (default) | `OPENAI_API_KEY` | paid |
| OpenRouter | `https://openrouter.ai/api/v1` | free key at openrouter.ai/keys | free `:free` models, no card |
| Ollama (local) | `http://localhost:11434/v1` | none needed | free, private |
| opencode Zen / other gateways | your gateway URL | your gateway key | varies |

## Configure

```bash
cp .env.example .env
# OpenRouter free example:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-or-v1-...
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# OPENAI_CHAT_MODEL=nvidia/nemotron-3.5-lightning:free
```

## Verify with one real call

```bash
python scripts/live_smoke.py
# provider=... model=... key=***1234 latency_ms=900
# LIVE_SMOKE=OK
```

The script makes exactly one chat-completions call, prints latency and
whether the answer kept its `[1]` citation, and never prints the key.
CI stays keyless: tests and evals always use the mock.

## Notes

- Free models vary in instruction-following; if citations drop, try a
  larger `:free` model or lower temperature (already 0.2 in prod path).
- Timeouts are 60s. Production hardening (retries with backoff on 429/5xx) is tracked future work.
- Never commit `.env` or keys — `.gitignore` excludes them.
