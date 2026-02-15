# main.py — MCP HTTP server exposing RAG tools
import contextlib

from fastapi import FastAPI
from mcp.server import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from config import settings
from query import run_query_with_chunks

# streamable_http_path="/" so mounted at /mcp matches (path becomes /)
mcp = FastMCP(
    settings.mcp_name,
    stateless_http=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)

@mcp.tool()
def rag_query_with_chunks(
    question: str,
    request_id: int | str | None = None,
    session_id: int | str | None = None,
):
    """Answer plus top ranked chunks. Optional request_id and session_id are sent as LangSmith tags."""
    result = run_query_with_chunks(
        question,
        request_id=request_id,
        session_id=session_id,
    )
    return {
        "metadata": result["metadata"],
        "error": None,
        "data": {
            "question": question,
            "answer": result["answer"],
        },
    }


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