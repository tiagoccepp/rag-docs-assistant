#!/usr/bin/env python3
"""Live smoke: ONE real call to an OpenAI-compatible LLM API (RAG generation).

Reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_CHAT_MODEL from env (see .env.example).
Refuses to run without a key (except localhost endpoints like Ollama).
Never prints the key. Exits nonzero on HTTP/API errors.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
LOCAL = "localhost" in BASE or "127.0.0.1" in BASE

if not KEY and not LOCAL:
    print("Missing OPENAI_API_KEY. See docs/PROVIDERS.md for free options (OpenRouter :free, Ollama).")
    sys.exit(2)

chunk = (Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "01-handbook.md").read_text()[:1200]
payload = {
    "model": MODEL,
    "temperature": 0.2,
    "max_tokens": 1024,  # reasoning models spend tokens thinking before answering
    "messages": [
        {"role": "system", "content": "Answer using ONLY the context. Cite facts as [1]. If unanswerable, say exactly: I do not know."},
        {"role": "user", "content": f"[1]\n{chunk}\n\nQuestion: What are the core collaboration hours?"},
    ],
}
req = urllib.request.Request(
    f"{BASE}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}", "User-Agent": "portfolio-live-smoke/1.0", "x-opencode-session": os.getenv("OPENCODE_SESSION_ID", "live-smoke")},  # Zen API requires the session header; ignored by OpenAI/OpenRouter
)
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.load(r)
except Exception as e:  # noqa: BLE001 - smoke script, surface the error
    print(f"LIVE_SMOKE=ERROR {type(e).__name__}: {e}")
    sys.exit(1)
dt = (time.time() - t0) * 1000
answer = (body["choices"][0]["message"].get("content") or "").strip()
print(f"provider={BASE} model={MODEL} key=***{KEY[-4:] if KEY else 'none'} latency_ms={dt:.0f}")
print(f"cited={'[1]' in answer}")
print("--- answer ---")
print(answer[:800])
print("LIVE_SMOKE=OK")
