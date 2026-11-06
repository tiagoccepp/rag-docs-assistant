# Deploy

Primary target: **Hugging Face Spaces (Docker mode)** — free, no card, persistent URL.

```bash
HF_TOKEN=hf_xxx python scripts/hf_deploy.py   # creates/updates the Space, prints the URL
```

Then open `https://<you>-rag-docs.hf.space/ui`.

## How it works

- `Dockerfile` serves the API with uvicorn on `${PORT:-7860}` (Spaces sets 7860;
  Render-style `$PORT` also works).
- The deploy script uploads only `Dockerfile`, `requirements.txt`, `src/`, `data/`
  plus a generated Space `README.md` (SDK header).
- No env vars required: the app boots on the offline mock backend. Optional live
  LLM: set `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`, `OPENAI_CHAT_MODEL`) as Space secrets.

## Alternatives

- **Render/Railway**: connect the GitHub repo, Docker runtime, free tier. Same image.
- **Local**: `./demo.sh` (port 8000) — identical UI.
