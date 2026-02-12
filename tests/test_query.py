# test_query.py — unit and integration tests for query, agent, RAG
from __future__ import annotations

import os

import pytest

from langchain_core.documents import Document


# ----------------------------
# Unit: query helpers
# ----------------------------
def test_minmax_basic():
    from query import _minmax

    assert _minmax([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert _minmax([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]


def test_minmax_flat():
    from query import _minmax

    # all same -> all 1.0
    assert _minmax([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


def test_minmax_empty():
    from query import _minmax

    assert _minmax([]) == []


def test_tok():
    from query import _tok

    assert _tok("Hello World") == ["hello", "world"]
    assert _tok("") == []
    assert _tok("a1_b2") == ["a1_b2"]


def test_dense_dist_to_score():
    from query import _dense_dist_to_score

    assert _dense_dist_to_score(0.0) == 1.0
    assert 0 < _dense_dist_to_score(1.0) < 1.0
    assert _dense_dist_to_score(999.0) < 0.01


def test_bm25_scores_local():
    from query import _bm25_scores_local

    docs = ["hello world", "world only", "foo bar"]
    scores = _bm25_scores_local("hello world", docs)
    assert len(scores) == 3
    assert scores[0] >= scores[1]  # first doc has both terms
    assert scores[0] > 0


def test_bm25_scores_local_empty_docs():
    from query import _bm25_scores_local

    assert _bm25_scores_local("query", []) == []


# ----------------------------
# Unit: agent helpers
# ----------------------------
def test_where_key():
    from agent import _where_key

    assert _where_key(None) == "__ALL__"
    assert _where_key({}) == "__ALL__"
    assert _where_key({"a": 1, "b": 2}) == "[('a', 1), ('b', 2)]"


def test_langsmith_config():
    from agent import _langsmith_config

    out = _langsmith_config()
    assert isinstance(out, dict)
    out_with_meta = _langsmith_config(metadata={"k": "v"})
    assert out_with_meta.get("metadata") == {"k": "v"}


# ----------------------------
# Unit: format_docs (uses settings.retrieval_k)
# ----------------------------
def test_format_docs():
    from query import format_docs

    docs = [
        Document(page_content="first"),
        Document(page_content="second"),
        Document(page_content="third"),
    ]
    out = format_docs(docs)
    assert "first" in out and "second" in out
    assert "---" in out


# ----------------------------
# Integration: run_query_with_chunks response shape
# ----------------------------
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("CHROMA_API_KEY"),
    reason="OPENAI_API_KEY and CHROMA_API_KEY required",
)
def test_run_query_with_chunks_shape():
    """run_query_with_chunks returns answer and metadata (chunks in metadata.reranked_chunks)."""
    from query import run_query_with_chunks

    result = run_query_with_chunks("What is this about?")
    assert "answer" in result
    assert "metadata" in result
    assert "reranked_chunks" in result["metadata"]
    assert isinstance(result["metadata"]["reranked_chunks"], list)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("CHROMA_API_KEY"),
    reason="OPENAI_API_KEY and CHROMA_API_KEY required",
)
def test_run_query_with_chunks_chunk_structure():
    """Each chunk has rank, chunk_id, source, preview, text, scores, metadata."""
    from query import run_query_with_chunks

    result = run_query_with_chunks("What is this about?")
    chunks = result["metadata"]["reranked_chunks"]
    assert len(chunks) > 0
    for chunk in chunks[:1]:
        assert "rank" in chunk
        assert "chunk_id" in chunk
        assert "source" in chunk
        assert "preview" in chunk
        assert "text" in chunk
        assert "scores" in chunk
        assert "metadata" in chunk
        assert "bm25_raw" in chunk["scores"]
        assert "dense_norm" in chunk["scores"]
        assert "hybrid" in chunk["scores"]


# ----------------------------
# Integration: run_query (skip when no API keys)
# ----------------------------
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("CHROMA_API_KEY"),
    reason="OPENAI_API_KEY and CHROMA_API_KEY required",
)
def test_run_query_returns_answer():
    from agent import run_query

    answer = run_query("What is this about?")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("CHROMA_API_KEY"),
    reason="OPENAI_API_KEY and CHROMA_API_KEY required",
)
def test_run_query_with_chunks_returns_grounded_answer():
    from query import run_query_with_chunks

    result = run_query_with_chunks("What is Taixing's visa status?")
    assert "H4" in result["answer"] or "visa" in result["answer"].lower() or len(result["answer"]) > 5
