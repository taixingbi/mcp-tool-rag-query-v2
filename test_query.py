"""Integration tests for RAG query pipeline."""
import pytest

from query import run_query, run_query_with_chunks


def test_run_query_returns_answer():
    """RAG pipeline returns a non-empty string response."""
    answer = run_query("What is this about?")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


def test_run_query_compensation_returns_grounded_answer():
    """When corpus has compensation data, answer should reference it or state no context."""
    answer = run_query("what compensation")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    # Either cites salary/compensation from context, or states context lacks it
    has_salary_info = any(
        word in answer.lower() for word in ["salary", "compensation", "180", "240", "000"]
    )
    has_no_context = "does not" in answer.lower() or "no information" in answer.lower()
    assert has_salary_info or has_no_context


def test_run_query_visa_returns_grounded_answer():
    """When corpus has visa data, answer should reference it or state no context."""
    answer = run_query("what visa status")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    has_visa_info = any(
        word in answer.lower() for word in ["visa", "h4", "ead", "sponsorship"]
    )
    has_no_context = "does not" in answer.lower() or "no information" in answer.lower()
    assert has_visa_info or has_no_context


def test_run_query_with_chunks_returns_answer_and_chunks():
    """run_query_with_chunks returns answer, chunks, and metadata."""
    result = run_query_with_chunks("What is this about?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "chunks" in result
    assert "metadata" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0
    assert isinstance(result["chunks"], list)
    assert isinstance(result["metadata"], dict)
    assert "reranked_chunks" in result["metadata"]
    assert isinstance(result["metadata"]["reranked_chunks"], list)
    assert len(result["chunks"]) == len(result["metadata"]["reranked_chunks"])


def test_run_query_with_chunks_chunks_have_expected_structure():
    """Chunks returned by run_query_with_chunks have expected fields."""
    result = run_query_with_chunks("What is this about?")
    assert len(result["chunks"]) > 0
    chunk = result["chunks"][0]
    assert isinstance(chunk, dict)
    assert "chunk_id" in chunk
    assert "distance" in chunk
    assert "text" in chunk
    assert "metadata" in chunk
    assert isinstance(chunk["distance"], (int, float))
    assert isinstance(chunk["text"], str)
