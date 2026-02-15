# agent.py — RAG prompt, chain build, run_query, LangSmith config
from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings

# ----------------------------
# Chain cache + where key
# ----------------------------
_rag_chain_cache: Dict[str, Any] = {}


def _where_key(where: Optional[Dict[str, Any]]) -> str:
    if not where:
        return "__ALL__"
    try:
        return str(sorted(where.items()))
    except Exception:
        return str(where)


# ----------------------------
# Prompt + chain
# ----------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions based only on the provided context. "
            "If the context does not contain relevant information, say so. "
            "Do not make up facts. Cite the context when possible.",
        ),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


def build_rag_chain(where: Optional[Dict[str, Any]] = None):
    """
    Build RAG chain. Cached per-where (IMPORTANT for correctness if where changes).
    """
    from query import format_docs, get_retriever

    key = _where_key(where)
    if key not in _rag_chain_cache:
        _rag_chain_cache[key] = (
            {"context": get_retriever(where=where) | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | ChatOpenAI(
                model=settings.openai_model,
                temperature=0,
                timeout=30,
                max_retries=2,
            )
            | StrOutputParser()
        )
    return _rag_chain_cache[key]


def run_query(
    question: str,
    where: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[Any] = None,
    session_id: Optional[Any] = None,
) -> str:
    tags = []
    if getattr(settings, "app_version", None):
        tags.append(f"app_version:{settings.app_version}")
    if getattr(settings, "mcp_name", None):
        tags.append(f"mcp_name:{settings.mcp_name}")
    if request_id is not None:
        tags.append(f"request_id:{request_id}")
    if session_id is not None:
        tags.append(f"session_id:{session_id}")
    config = {"run_name": settings.mcp_name, "tags": tags}
    if metadata:
        config["metadata"] = metadata
    return build_rag_chain(where=where).invoke(question, config=config)
