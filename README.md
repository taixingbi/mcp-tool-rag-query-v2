# RAG Pipeline

RAG over documents using **Chroma Cloud** (dense search) and **LangChain**. Exposes tools via **MCP** (Model Context Protocol) so clients can call `rag_query` and `rag_query_with_chunks` over HTTP.

---

## Setup

<!-- Install Python deps in a venv before running or deploying. -->
Create a virtualenv and install dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

<!-- .env is not committed; copy from .env.example or set these keys. -->
Create a `.env` file in the project root:

```
APP_VERSION=v:1.01
OPENAI_API_KEY=your-openai-key
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=rag_dev
```

---

## Local Development

<!-- All commands below assume you are in the repo root and have activated the venv. -->

### Run RAG from CLI

```bash
python query.py "what is taixing visa"
```

### Run MCP HTTP Server

```bash
uvicorn mcp_server:app --reload --port 8000
```

### Test Endpoints

**Health check:**

```bash
curl http://127.0.0.1:8000/health
```

**Call MCP tools** (use trailing slash `/mcp/` to avoid 307 redirect):

**`rag_query`** — returns only the RAG answer:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  http://localhost:8000/mcp/
```

**`rag_query_with_chunks`** — returns answer + ranked chunks as JSON:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  http://localhost:8000/mcp/
```

Response is JSON-RPC; the tool result is a JSON string with: `answer`, `chunks` (each with `rank`, `chunk_id`, `source`, `preview`, `text`, `scores` (e.g. `bm25_raw`, `bm25_norm`, `dense_raw`, `dense_norm`, `distance`, `hybrid`), `metadata`), `used_chunk_ids` (unique chunk IDs used for the answer), `retrieval` (`k`, `alpha`, `filters`, `warnings`).

---

## Docker

<!-- Image does not bundle .env; pass --env-file at run time. -->
Build and run:

```bash
docker build -t rag-mcp .
docker run -p 8000:8000 --env-file .env rag-mcp
```

> **Note:** If you see *"api_key client option must be set"*, ensure `--env-file .env` points to the file containing `OPENAI_API_KEY`.

---

## Fly.io Deployment

<!-- One Fly app per env; set secrets per app; deploy with --app <name>. -->
Use one app per environment: `mcp-tool-rag-query-v2-{env}` where `{env}` = `dev`, `qa`, or `prod`. Each app gets its own secrets; the same `fly.toml` is used for all.

### One-time Setup

```bash
brew install flyctl
fly auth login
fly auth token  # Use output as GitHub Actions secret FLY_API_TOKEN if using CI
```

### Create Apps

Run once per environment:

```bash
fly launch --name mcp-tool-rag-query-v2-dev
fly launch --name mcp-tool-rag-query-v2-qa
fly launch --name mcp-tool-rag-query-v2-prod
```

### Set Secrets

<!-- Deploy: use --app to target dev/qa/prod; omit --no-cache for faster builds when deps unchanged. -->
```bash
fly deploy --no-cache --app mcp-tool-rag-query-v2-dev
fly deploy --no-cache --app mcp-tool-rag-query-v2-qa
fly deploy --no-cache --app mcp-tool-rag-query-v2-prod
```

**QA** (from `.env.qa`):

```bash
curl https://mcp-tool-rag-query-v2-dev.fly.dev/health
```

**Prod** (from `.env.prod`):

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-dev.fly.dev/mcp/
```

> **Tip:** If `grep`/`cut` corrupts keys (e.g., 401 errors), paste the key directly: `fly secrets set -a mcp-tool-rag-query-v2-dev OPENAI_API_KEY="sk-proj-YOUR_KEY"`

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-dev.fly.dev/mcp/
```

### Test Deployed Apps

Replace `{env}` with `dev`, `qa`, or `prod`:

**Health check:**

```bash
curl https://mcp-tool-rag-query-v2-dev.fly.dev/health
```

**Call MCP tools:**

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-dev.fly.dev/mcp/
```

# rag_query_with_chunks
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-dev.fly.dev/mcp/
```

=======

<!-- .env is not committed; copy from .env.example or set these keys. -->
Create a `.env` file in the project root:

```
APP_VERSION=v:1.01
OPENAI_API_KEY=your-openai-key
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=rag_dev
```

---

## Local run

<!-- All commands below assume you are in the repo root and have activated the venv. -->
### Run RAG from the CLI (no server)

```bash
python query.py "what is taixing visa"
```

### Run the MCP HTTP server

```bash
uvicorn mcp_server:app --reload --port 8000
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Call MCP tools via curl

<!-- Use trailing slash on /mcp/ to avoid 307 redirect from the framework. -->
Use **trailing slash** (`/mcp/`) to avoid 307 redirect.

**`rag_query`** — returns only the RAG answer (plain text):

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  http://localhost:8000/mcp/
```

**`rag_query_with_chunks`** — returns answer + ranked chunks as JSON:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  http://localhost:8000/mcp/
```


### Docker

<!-- Image does not bundle .env; pass --env-file at run time. -->
Build the image:

```bash
docker build -t rag-mcp .
```

Run the container with env from `.env`. The image does not include env files (they are in `.dockerignore`).

```bash
docker run -p 8000:8000 --env-file .env rag-mcp
```

If you see *"api_key client option must be set"*, the container is not getting `OPENAI_API_KEY`. Use `--env-file .env` from the directory that contains that file.


### Fly.io (dev / qa / prod)

<!-- One Fly app per env; set secrets per app; deploy with --app <name>. -->
Use one app per environment: `mcp-tool-rag-query-v2-{env}` with `{env}` = `dev`, `qa`, or `prod`. Each app gets its own secrets from the matching env file; the same `fly.toml` is used for all.

**One-time setup** (run from repo root):

```bash
brew install flyctl
fly auth login
fly auth token
```

Use `fly auth token` output as GitHub Actions secret `FLY_API_TOKEN` if you use CI.

**Create apps per environment** (run once per env; each creates an app and uses the repo’s `fly.toml`):

```bash
fly launch --name mcp-tool-rag-query-v2-dev
fly launch --name mcp-tool-rag-query-v2-qa
fly launch --name mcp-tool-rag-query-v2-prod
```


<!-- Deploy: use --app to target dev/qa/prod; omit --no-cache for faster builds when deps unchanged. -->
```bash
fly deploy --no-cache --app mcp-tool-rag-query-v2-dev
fly deploy --no-cache --app mcp-tool-rag-query-v2-qa
fly deploy --no-cache --app mcp-tool-rag-query-v2-prod
```

** dev

<!-- Verify app is up before calling /mcp/. -->
```bash
curl https://mcp-tool-rag-query-v2-dev.fly.dev/health
```

**`rag_query`** — returns only the RAG answer (plain text):
```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-dev.fly.dev/mcp/
```

**`rag_query_with_chunks`** — answer plus ranked chunks as JSON:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-dev.fly.dev/mcp/
```

** qa

<!-- Verify app is up before calling /mcp/. -->
```bash
curl https://mcp-tool-rag-query-v2-qa.fly.dev/health
```

**`rag_query`** — returns only the RAG answer (plain text):

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-qa.fly.dev/mcp/
```

**`rag_query_with_chunks`** — answer plus ranked chunks as JSON:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-qa.fly.dev/mcp/
```

** prod

<!-- Verify app is up before calling /mcp/. -->
```bash
curl https://mcp-tool-rag-query-v2-prod.fly.dev/health
```

**`rag_query`** — returns only the RAG answer (plain text):

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-prod.fly.dev/mcp/
```

**`rag_query_with_chunks`** — answer plus ranked chunks as JSON:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag_query_with_chunks","arguments":{"question":"what is Taixing visa?"}},"id":1}' \
  https://mcp-tool-rag-query-v2-prod.fly.dev/mcp/
```