# query.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import numpy as np

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi

from config import settings
from chroma_client import get_chroma_collection
from agent import run_query


# ----------------------------
# Lazy singletons
# ----------------------------
_embedder = None


def _get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=settings.embedding_model)
    return _embedder


# ----------------------------
# Dense search
# ----------------------------
def _search_dense(
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Dense search:
      1) embed query using OpenAI embeddings
      2) query Chroma with query_embeddings
    """
    coll = get_chroma_collection()
    q_emb = _get_embedder().embed_query(query)

    res = coll.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,  # e.g. {"tenant_id": "t1"} (optional)
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[dict] = []
    for i in range(len(docs)):
        meta = metas[i] or {}
        dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 0.0
        out.append(
            {
                "chunk_id": meta.get("chunk_id", ""),
                "distance": dist,          # smaller = better
                "combined_score": dist,    # backward compatible name
                "dense_score": dist,       # backward compatible name (distance)
                "bm25_score": 0.0,
                "text": docs[i] or "",
                "metadata": meta,
            }
        )
    return out


# ----------------------------
# Hybrid search: dense top-N + BM25 rerank on those N only
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def _minmax(vals: List[float]) -> List[float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return []
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.ones_like(arr).tolist()
    return ((arr - mn) / (mx - mn)).tolist()


def _dense_dist_to_score(dist: float) -> float:
    # smaller distance is better -> convert to larger-better score
    return 1.0 / (1.0 + max(dist, 0.0))


def _bm25_scores_local(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """
    BM25 computed on a small candidate list (e.g. top-N from dense).
    Uses rank_bm25 (BM25Okapi) for standard, well-tested scoring.
    """
    if not docs:
        return [0.0] * len(docs)

    doc_tokens = [_tok(d) for d in docs]
    q_terms = _tok(query)

    bm25 = BM25Okapi(doc_tokens, k1=k1, b=b)
    scores = bm25.get_scores(q_terms)
    return [float(s) for s in scores]


def _search_hybrid(
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
    *,
    dense_candidates: Optional[int] = None,
    alpha: Optional[float] = None,
) -> List[dict]:
    """
    Hybrid retrieval (startup-friendly):
      1) dense recall: top-N from Chroma
      2) local BM25 rerank over those N
      3) weighted fuse: alpha * dense + (1-alpha) * bm25
    """
    # configurable via env/settings; fallback defaults
    dense_candidates = int(
        dense_candidates
        if dense_candidates is not None
        else getattr(settings, "hybrid_candidates", 200)
    )
    alpha = float(alpha if alpha is not None else getattr(settings, "hybrid_alpha", 0.7))

    dense_hits = _search_dense(query, k=dense_candidates, where=where)

    texts = [h.get("text", "") for h in dense_hits]

    dense_raw = [_dense_dist_to_score(float(h.get("distance", 999999.0))) for h in dense_hits]
    dense_norm = _minmax(dense_raw)

    bm25_raw = _bm25_scores_local(query, texts)
    # When all BM25 are 0, use zeros for bm25_norm so hybrid = alpha*dense only (no fake 1.0)
    bm25_norm = (
        [0.0] * len(bm25_raw)
        if (max(bm25_raw) if bm25_raw else 0.0) < 1e-12
        else _minmax(bm25_raw)
    )

    for i, h in enumerate(dense_hits):
        h["dense_raw_score"] = dense_raw[i]
        h["dense_norm_score"] = dense_norm[i]
        h["bm25_raw_score"] = bm25_raw[i]
        h["bm25_norm_score"] = bm25_norm[i]
        h["hybrid_score"] = alpha * dense_norm[i] + (1.0 - alpha) * bm25_norm[i]
        h["combined_score"] = h["hybrid_score"]  # keep backward-compatible name

    dense_hits.sort(key=lambda x: float(x.get("hybrid_score", 0.0)), reverse=True)
    return dense_hits[:k]


# ----------------------------
# Retriever
# ----------------------------
class CloudRetriever(BaseRetriever):
    """
    Retriever using hybrid search (dense top-N + BM25 rerank on those N).
    """
    where: Optional[Dict[str, Any]] = None  # you can set this externally

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        hits = _search_hybrid(query, k=settings.retrieval_k * 2, where=self.where)
        return [Document(page_content=h["text"], metadata=h["metadata"]) for h in hits]


def get_retriever(where: Optional[Dict[str, Any]] = None) -> CloudRetriever:
    r = CloudRetriever()
    r.where = where
    return r


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs[:settings.retrieval_k])


# ----------------------------
# Return ranked chunks + answer
# ----------------------------
def retrieve_ranked_chunks(
    question: str,
    k: int = settings.retrieval_k * 2,
    where: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Return ranked chunks for debugging / UI.
    Each chunk has rank (1-based), scores (nested), source, preview, text, metadata.
    """
    hits = _search_hybrid(question, k=k, where=where)

    ranked: List[dict] = []
    for rank_1based, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        text = h.get("text") or ""
        chunk_id = h.get("chunk_id", "") or meta.get("chunk_id", "")
        ranked.append(
            {
                "rank": rank_1based,
                "chunk_id": chunk_id,
                "source": meta.get("source", ""),
                "preview": text[:250],
                "text": text,
                "scores": {
                    "bm25_raw": h.get("bm25_raw_score", 0.0),
                    "bm25_norm": h.get("bm25_norm_score", 0.0),
                    "dense_raw": h.get("dense_raw_score", 0.0),
                    "dense_norm": h.get("dense_norm_score", 0.0),
                    "distance": h.get("distance", 0.0),
                    "hybrid": h.get("hybrid_score", 0.0),
                },
                "metadata": meta,
            }
        )
    return ranked


def run_query_with_chunks(
    question: str,
    where: Optional[Dict[str, Any]] = None,
    chunk_k: int = settings.retrieval_k * 2,
) -> Dict[str, Any]:
    chunks = retrieve_ranked_chunks(question, k=chunk_k, where=where)
    answer = run_query(
        question,
        where=where,
        metadata={"reranked_chunks": chunks},
    )

    return {
        "answer": answer,
        "metadata": {"reranked_chunks": chunks},
    }
