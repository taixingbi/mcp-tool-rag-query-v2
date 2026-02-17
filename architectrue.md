# Architecture: RAG MCP Tool Server

An LLM/RAG architecture plan for `taixingbi/mcp-tool-rag-query-v2`, based directly on the code + README.

---

## 1. What This Service Is

A RAG MCP tool server that exposes a RAG pipeline over MCP (Model Context Protocol) "streamable HTTP", mounted at `/mcp` inside a FastAPI app. It provides at least one tool:

```
rag_query_with_chunks(question, request_id?, session_id?) → returns final answer + ranked chunks + retrieval metadata
```

The service is designed so a client (orchestrator / frontend) can call it via JSON-RPC to get both:

- **LLM answer** — the generated response
- **Explainability** — ranked evidence chunks + per-chunk scores

---

## 2. High-Level Component Diagram

```mermaid
flowchart LR
  Client[Client: Orchestrator / Frontend] -->|JSON-RPC tools/call| API[FastAPI App]
  API -->|mounted /mcp| MCP[FastMCP Streamable HTTP App]

  MCP --> Tool[Tool: rag_query_with_chunks]
  Tool --> RAG[LangChain RAG Chain]

  subgraph Retrieval
    RAG --> Dense[Dense Recall: Chroma Cloud query_embeddings]
    Dense --> Candidates[Top-N dense candidates]
    Candidates --> BM25["Local BM25 rerank (rank_bm25)"]
    BM25 --> Fuse["Hybrid fuse: alpha*dense_norm + (1-alpha)*bm25_norm"]
  end

  Dense --> Chroma[(Chroma Cloud Collection)]
  RAG --> LLM[ChatOpenAI]
  RAG --> LS[LangSmith tags/metadata]
```

### Key Implementation Anchors

- FastAPI mounts MCP at `/mcp` and exposes `/health`
- Retrieval pipeline: dense → local BM25 rerank → weighted hybrid score
- The RAG chain is cached per `where` filter key to avoid correctness issues

---



### Why This Design Works Well

- **Stable tool interface** — MCP allows orchestrator to treat it like any other tool
- **Debuggable evidence** — `chunks[]` includes rank, source, preview/text, and score breakdown
- **Hybrid retrieval** — Improves precision without a second vector DB or external reranker service

---

## 3. Retrieval Architecture (The Core)

### Dense Recall (Chroma Cloud)

- Embeds the query using OpenAI embeddings
- Calls `collection.query(query_embeddings=[...], include=["documents","metadatas","distances"])`
- **Note:** The code sets `embedding_function=None` for the collection so Chroma won't auto-embed; embeddings are always passed explicitly

### Hybrid Scoring

1. Get top-N dense candidates (`hybrid_candidates`, default ~200 in code path)
2. Run BM25Okapi over those candidates only (fast, local)
3. Normalize scores and fuse:

   ```
   hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm
   ```

This is a startup-friendly hybrid: recall comes from dense search, precision from BM25 rerank, and cost stays low.

---

## 4. Generation Architecture (LLM Chain)

The LangChain chain structure:

```
context = retriever(where) | format_docs
```

- **Prompt:** Strict instruction — answer only from provided context; if missing, say so; don't make up facts
- **Model:** `ChatOpenAI(model=settings.openai_model, temperature=0, timeout=30, max_retries=2)`
- **Output:** `StrOutputParser()`

**Important:** The chain is cached per `where` key (`_rag_chain_cache` keyed by `_where_key(where)`), which prevents cross-tenant/filter leakage.

---

## 5. Observability Hooks (LangSmith)

- **Tags:** `app_version`, `mcp_name`, plus `request_id`, `session_id` when provided
- **Run config:** `run_name=settings.mcp_name` so traces group by service/tool
- **Metadata:** `metadata={"reranked_chunks": chunks}` is passed into trace config so evidence is attached to the run

---

## 6. Public API Surface

### Health

| Endpoint    | Description                                                |
|------------|------------------------------------------------------------|
| `GET /health` | Returns status plus key runtime config (mcp name/version + chroma info) |

### MCP Tool

`rag_query_with_chunks(...)` returns a payload shaped like:

| Field      | Description                          |
|-----------|--------------------------------------|
| `metadata` | `{ reranked_chunks: [...] }`        |
| `data`     | `{ question, answer }`              |
| (internal) | `chunks`, `used_chunk_ids`, `retrieval {k, alpha, filters, warnings}` |

---

## 7. README-Ready Architecture Summary

Copy-paste this section as-is for README or design docs:

### Service Overview

This repository implements a Retrieval-Augmented Generation (RAG) microservice exposed as an MCP (Model Context Protocol) tool server over HTTP. A FastAPI application mounts a FastMCP streamable HTTP app under `/mcp`, enabling JSON-RPC tool invocation from an orchestrator or frontend.

### Retrieval Pipeline

The service uses Chroma Cloud for dense recall. Each query is embedded using OpenAI embeddings, and the Chroma collection is queried via `query_embeddings`. To improve precision and reduce hallucinations, the service performs a lightweight hybrid retrieval strategy: it first retrieves top-N dense candidates, then applies a local BM25 reranker on those candidates only, and finally fuses normalized dense and BM25 signals using a weighted score (alpha).

### Generation Pipeline

Retrieved chunks are formatted into a context block and passed to a LangChain RAG prompt that instructs the model to answer strictly from provided context and to abstain when evidence is missing. The LLM call uses deterministic settings (temperature 0) with timeouts and retries.

### Explainability & Debugging

The main tool returns both the final answer and the ranked chunk list, including per-chunk score breakdown (dense distance, BM25 scores, normalized values, and fused hybrid score). The service also returns `used_chunk_ids` to indicate which evidence chunks were used in generation.

### Tracing / Observability

LangSmith tags (e.g., app version, MCP name, request/session IDs) and retrieval metadata are attached to each run for debugging and evaluation workflows.

---


## 8. Request/Response Lifecycle (End-to-End)

```mermaid
sequenceDiagram
  participant C as Client
  participant F as FastAPI (/mcp)
  participant M as MCP Tool (rag_query_with_chunks)
  participant R as Retriever (Hybrid)
  participant V as Chroma Cloud
  participant L as LLM (ChatOpenAI)

  C->>F: POST /mcp (JSON-RPC tools/call: rag_query_with_chunks)
  F->>M: dispatch tool call
  M->>R: retrieve_ranked_chunks(question, k=chunk_k)
  R->>V: query_embeddings (dense recall top-N)
  V-->>R: docs + metadatas + distances
  R->>R: BM25 rerank over top-N + hybrid fuse
  R-->>M: ranked chunks + scores
  M->>L: RAG prompt with formatted top-k context
  L-->>M: answer text
  M-->>C: { answer, chunks, used_chunk_ids, retrieval, metadata }
```