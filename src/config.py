"""Central configuration via Pydantic Settings.

All tunables are environment-driven with sane offline defaults.
Copy `.env.example` to `.env` and adjust.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from env / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = Field(default="postgresql://rag:ragpassword@localhost:5432/ragdocs")

    # Embeddings
    embedding_provider: Literal["local-hash", "sentence-transformers"] = "local-hash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # Retrieval
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 10
    top_k_final: int = 5
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Rerank
    rerank_provider: Literal["local", "cohere", "openai"] = "local"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"
    openai_api_key: str = ""
    openai_rerank_model: str = "gpt-4o-mini"

    # Generation
    llm_provider: Literal["mock", "openai"] = "mock"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.2
    abstain_threshold: float = 0.35
    min_context_docs: int = 1

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
