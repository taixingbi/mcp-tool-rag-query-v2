
```bash
curl https://mcp-tool-rag-query-v2-prod.fly.dev/health
```

**Call MCP tools:**

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-prod.fly.dev/mcp/
```

# rag_query_with_chunks
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-prod.fly.dev/mcp/
```