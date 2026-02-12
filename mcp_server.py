# mcp_server.py — MCP HTTP server exposing RAG tools
import contextlib
import json
from fastapi import FastAPI
from mcp.server import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from config import settings
from query import run_query, run_query_with_chunks

# streamable_http_path="/" so mounted at /mcp matches (path becomes /)
mcp = FastMCP(
    settings.mcp_name,
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        "mcp-tool-rag-query-v2-dev.fly.dev",
        "mcp-tool-rag-query-v2-qa.fly.dev",
        "mcp-tool-rag-query-v2-prod.fly.dev",
    ],
),
)

@mcp.tool()
def rag_query(question: str) -> str:
    """Answer a question using RAG (Chroma + LLM)."""
    return run_query(question)


@mcp.tool()
def rag_query_with_chunks(question: str) -> str:
    """Answer plus top ranked chunks as JSON."""
    return json.dumps(run_query_with_chunks(question), ensure_ascii=False)


mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(title=settings.mcp_name, version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    return {"status": "ok", 
            "mcp": settings.mcp_name, 
            "version": settings.app_version, 
            "LANGCHAIN_PROJECT": settings.langchain_project, 
            "CHROMA_DATABASE": settings.chroma_database, 
            "CHROMA_COLLECTION": settings.chroma_collection}

app.mount("/mcp", mcp_app)