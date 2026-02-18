# MemoryDB

Self-hosted persistent memory for AI agents. Knowledge graph, hybrid search, pluggable providers.

## What is MemoryDB?

MemoryDB gives AI agents long-term memory through a REST API backed by PostgreSQL and pgvector. You ingest content (text, URLs, PDFs, code, audio), and it automatically extracts structured memories, embeds them, and connects them in a knowledge graph. Search combines vector similarity, BM25 full-text, and graph traversal to return the most relevant results.

## Features

- **Knowledge graph memory model** -- memories connect through update, extend, and derive relationships
- **Hybrid search** -- vector similarity + BM25 full-text + graph traversal in a single query
- **Pluggable embedding providers** -- OpenAI, Ollama, or sentence-transformers
- **Pluggable LLM providers** -- OpenAI, Anthropic, or Ollama for memory extraction
- **Multi-agent support** -- container namespacing isolates memories per agent
- **Async processing pipeline** -- content extraction, chunking, embedding, and LLM analysis run in background tasks
- **Full content type support** -- text, URLs, PDFs, images, audio, code, CSV, JSON, DOCX, XLSX
- **Auto-profiles** -- generates agent/user profiles from accumulated memories
- **Contradiction detection** -- new facts that conflict with old ones automatically supersede them
- **One-command deployment** -- Docker Compose with PostgreSQL + pgvector included

## Quick start

```bash
git clone https://github.com/youruser/memorydb.git
cd memorydb
cp .env.example .env
# Edit .env -- set POSTGRES_PASSWORD and your API key(s)
docker compose up -d
```

The API is available at `http://localhost:8080`. Verify its running:

```bash
curl http://localhost:8080/health
```

## Configuration

All settings are controlled through environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DB` | `memorydb` | PostgreSQL database name |
| `POSTGRES_USER` | `memory` | PostgreSQL username |
| `POSTGRES_PASSWORD` | -- | PostgreSQL password (required) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port on host |
| `DATABASE_URL` | constructed | Full asyncpg connection string |
| `EMBEDDING_PROVIDER` | `openai` | Embedding backend: `openai`, `ollama`, `sentence-transformers` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name for embeddings |
| `LLM_PROVIDER` | `openai` | LLM backend: `openai`, `anthropic`, `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for memory extraction |
| `OPENAI_API_KEY` | -- | Required if using OpenAI providers |
| `ANTHROPIC_API_KEY` | -- | Required if using Anthropic LLM provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL for local providers |
| `API_PORT` | `8080` | Port the API listens on |
| `LOG_LEVEL` | `info` | Logging level: `debug`, `info`, `warning`, `error` |

## API reference

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check. Returns status and version. |
| `GET` | `/v1/stats` | Memory/document counts, embedding coverage, relationship totals. |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/documents` | Ingest content. Returns `202 Accepted` and queues processing. |
| `GET` | `/v1/documents/{id}` | Get document status and details. |
| `DELETE` | `/v1/documents/{id}` | Delete document and its associated memories. |
| `POST` | `/v1/documents/list` | List documents with optional container/status filtering. |

### Memories

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/memories/{id}` | Get a specific memory by ID. |
| `DELETE` | `/v1/memories/{id}` | Delete a memory. |
| `GET` | `/v1/memories/{id}/related` | Get related memories via graph edges. Optional `relationship_type` filter. |
| `PATCH` | `/v1/memories/{id}` | Update memory metadata or tags. |

### Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/search` | Hybrid search across memories. Supports container filtering, pagination, and related memory inclusion. |

### Containers

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/containers` | List all containers. Optional `prefix` filter. |
| `DELETE` | `/v1/containers/{name}` | Delete a container and all its memories. Requires `confirm=true`. |

### Profile

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/profile/{container}` | Get auto-generated profile for a container. Optional `fact_limit` and `recent_limit` params. |

## Usage examples

**Ingest text content:**

```bash
curl -X POST http://localhost:8080/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "We decided to use PostgreSQL for the main database. Deploy target is Docker on AWS ECS.",
    "container": "my-agent",
    "content_type": "text",
    "metadata": {"source": "meeting-notes"},
    "tags": ["decisions", "infrastructure"]
  }'
```

Response (`202 Accepted`):

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "container": "my-agent",
  "content_type": "text",
  "status": "queued",
  "created_at": "2026-02-18T12:00:00Z"
}
```

**Search memories:**

```bash
curl -X POST http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what database are we using",
    "container": "my-agent",
    "limit": 5,
    "include_related": true
  }'
```

Response:

```json
{
  "query": "what database are we using",
  "results": [
    {
      "id": "...",
      "container": "my-agent",
      "content": "Team decided to use PostgreSQL for the main database",
      "memory_type": "fact",
      "similarity": 0.94,
      "tags": ["decisions", "infrastructure"],
      "created_at": "2026-02-18T12:00:00Z",
      "related": [
        {"id": "...", "content": "Deploy target is Docker on AWS ECS", "relationship": "extends"}
      ]
    }
  ],
  "total": 1,
  "limit": 5,
  "offset": 0
}
```

**Get a container profile:**

```bash
curl http://localhost:8080/v1/profile/my-agent?fact_limit=10&recent_limit=5
```

**Check system stats:**

```bash
curl http://localhost:8080/v1/stats
```

## Architecture

MemoryDB processes content through an async pipeline with five stages:

```
Document --> Extract --> Chunk --> Embed --> LLM Extract --> Graph
  queued   extracting  chunking  embedding    indexing       done
```

1. **Extract** -- Pull text from the source. Plain text passes through directly. URLs get converted from HTML to markdown. PDFs, images, and audio go through their respective parsers.

2. **Chunk** -- Split extracted text into ~500 token pieces. The chunker is content-aware: markdown splits on headings, code splits on function/class boundaries, transcripts split on speaker turns.

3. **Embed** -- Generate vector embeddings for each chunk using the configured provider (OpenAI, Ollama, or sentence-transformers).

4. **LLM Extract** -- An LLM reads each chunk set and extracts structured memories: facts, preferences, episodes, insights, and tasks. It assigns confidence scores, detects contradictions with existing memories, and identifies temporal expiration.

5. **Graph Build** -- Extracted memories are linked to existing ones through three relationship types:
   - **updates** -- new memory supersedes an old one (marks old as not latest)
   - **extends** -- new memory adds detail to an existing one (both remain active)
   - **derives** -- system-inferred connection across memories

Search combines all three storage layers. Vector similarity finds semantically close memories. BM25 handles keyword matches. Graph traversal pulls in related memories that wouldn't surface through text matching alone. The scores get fused into a single ranked result list.

### Storage

PostgreSQL with pgvector handles everything in one database:

- `documents` -- raw ingested content and processing status
- `memories` -- extracted knowledge units with vector embeddings
- `memory_relationships` -- graph edges connecting memories
- `containers` -- agent namespace tracking

Indexes include ivfflat for vector search, GIN for tags and full-text, and B-tree for container/type/date filtering.

## License

MIT
