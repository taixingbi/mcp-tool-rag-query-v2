# chroma_client.py — Chroma Cloud client and collection (lazy singleton)
from __future__ import annotations

from config import settings

_collection = None


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


def get_chroma_collection():
    """
    Chroma Cloud collection (cached).
    embedding_function=None so we always pass query_embeddings / embeddings explicitly.
    """
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_collection(
            name=settings.chroma_collection,
            embedding_function=None,
        )
    return _collection
