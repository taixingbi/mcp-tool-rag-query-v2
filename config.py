# config.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env", override=True)


class Settings(BaseSettings):
    """All config from .env; add new vars here with Field(..., alias="ENV_NAME")."""
    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_version: str = Field(default="", alias="APP_VERSION")
    mcp_name: str = Field(default="mcp-rag-query-v2", alias="MCP_NAME")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langchain_endpoint: str | None = Field(default=None, alias="LANGCHAIN_ENDPOINT")
    langchain_project: str | None = Field(default=None, alias="LANGCHAIN_PROJECT")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    retrieval_k: int = Field(default=4, alias="RETRIEVAL_K")
    chroma_api_key: str | None = Field(default=None, alias="CHROMA_API_KEY")
    chroma_tenant: str | None = Field(default=None, alias="CHROMA_TENANT")
    chroma_database: str | None = Field(default=None, alias="CHROMA_DATABASE")
    chroma_collection: str = Field(default="taixing_identity", alias="CHROMA_COLLECTION")
    hybrid_candidates: int = Field(default=200, alias="HYBRID_CANDIDATES")
    hybrid_alpha: float = Field(default=0.7, alias="HYBRID_ALPHA")

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

def get_chroma_client():
    if not (settings.chroma_api_key and settings.chroma_tenant and settings.chroma_database):
        raise RuntimeError(
            "Missing CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE. Check .env."
        )
    import chromadb
    return chromadb.CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )
