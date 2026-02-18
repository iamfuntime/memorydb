# MemoryDB

Self-hosted persistent memory for AI agents. Knowledge graph, hybrid search, pluggable providers.

## What is MemoryDB?

MemoryDB gives AI agents long-term memory through a REST API backed by PostgreSQL and pgvector. You ingest content (text, URLs, PDFs, code, audio), and it automatically extracts structured memories, embeds them, and connects them in a knowledge graph. Search combines vector similarity, BM25 full-text, and graph traversal to return the most relevant results.

Ships with an [OpenClaw plugin](#openclaw-plugin) for drop-in agent memory: auto-recall, auto-capture, conversation logging, and session-start profile loading across every channel.

## Features

- **Knowledge graph memory model** -- memories connect through update, extend, and derive relationships
- **Hybrid search** -- vector similarity + BM25 full-text + graph traversal in a single query
- **Container inheritance** -- dot-notation namespaces (`team.agent1`) with hierarchical search across parent containers
- **Pluggable embedding providers** -- OpenAI, Ollama, or sentence-transformers
- **Pluggable LLM providers** -- OpenAI, Anthropic, or Ollama for memory extraction
- **Multi-agent support** -- container namespacing isolates memories per agent, team, or user
- **Async processing pipeline** -- content extraction, chunking, embedding, and LLM analysis run in background tasks
- **Full content type support** -- text, URLs, PDFs, images, audio, code, CSV, JSON, DOCX, XLSX
- **Auto-profiles** -- generates agent/user profiles from accumulated memories
- **Contradiction detection** -- new facts that conflict with old ones automatically supersede them
- **One-command deployment** -- Docker Compose with PostgreSQL + pgvector included
- **OpenClaw plugin** -- conversation logging, session profiles, auto-recall, auto-capture out of the box

---

## Quick start

```bash
git clone https://github.com/youruser/memorydb.git
cd memorydb
cp .env.example .env
# Edit .env -- set POSTGRES_PASSWORD and your API key(s)
docker compose up -d
```

The API is available at `http://localhost:8080`. Verify it's running:

```bash
curl http://localhost:8080/health
```

---

## Configuration

All settings are controlled through environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DB` | `memorydb` | PostgreSQL database name |
| `POSTGRES_USER` | `memory` | PostgreSQL username |
| `POSTGRES_PASSWORD` | -- | PostgreSQL password (required) |
| `DATABASE_URL` | constructed | Full asyncpg connection string |
| `EMBEDDING_PROVIDER` | `openai` | Embedding backend: `openai`, `ollama`, `sentence-transformers` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name for embeddings |
| `LLM_PROVIDER` | `openai` | LLM backend: `openai`, `anthropic`, `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for memory extraction |
| `OPENAI_API_KEY` | -- | Required if using OpenAI providers |
| `ANTHROPIC_API_KEY` | -- | Required if using Anthropic LLM provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL for local providers |
| `API_PORT` | `8080` | Port the API listens on |
| `API_BIND` | `127.0.0.1` | Bind address: `127.0.0.1` (localhost only) or `0.0.0.0` (all interfaces) |
| `API_TOKEN` | -- | Bearer token for API authentication (empty = no auth) |
| `LOG_LEVEL` | `info` | Logging level: `debug`, `info`, `warning`, `error` |

### Running fully local (no API keys)

Set both providers to Ollama and run a local model:

```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Without an LLM provider configured, the pipeline still works -- it produces chunk-level embeddings for vector search but skips structured memory extraction and relationship classification. Without an embedding provider, search falls back to BM25-only.

---

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
| `POST` | `/v1/search` | Hybrid search across memories. Supports container filtering, inheritance, pagination, and related memory inclusion. |

**Search request body:**

```json
{
  "query": "what database are we using",
  "container": "my-agent",
  "inherit": true,
  "limit": 10,
  "offset": 0,
  "include_related": true,
  "filters": {}
}
```

When `inherit` is `true` and the container uses dot-notation (e.g., `team.agent1`), search also includes results from parent containers (`team`).

### Containers

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/containers` | List all containers. Optional `prefix` filter. |
| `DELETE` | `/v1/containers/{name}` | Delete a container and all its memories. Requires `confirm=true`. |

### Profile

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/profile/{container}` | Get auto-generated profile for a container. Optional `fact_limit` and `recent_limit` params. |

Returns high-confidence facts/preferences (`static_facts`) and last-7-days activity (`recent_context`). Designed to be called at session start so an agent can load its context quickly.

---

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

**Ingest a URL:**

```bash
curl -X POST http://localhost:8080/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "https://example.com/blog/architecture-decisions",
    "container": "my-agent",
    "content_type": "url"
  }'
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

**Search with container inheritance:**

```bash
curl -X POST http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment process",
    "container": "team.agent1",
    "inherit": true,
    "limit": 10
  }'
```

This searches `team.agent1` first, then includes results from the `team` parent container.

**Get a container profile:**

```bash
curl http://localhost:8080/v1/profile/my-agent?fact_limit=10&recent_limit=5
```

**Check system stats:**

```bash
curl http://localhost:8080/v1/stats
```

---

## Architecture

### Processing pipeline

Every document ingested via `POST /v1/documents` runs through an async background pipeline:

```
Document --> Extract --> Chunk --> Embed --> LLM Extract --> Graph Build
  queued   extracting  chunking  embedding    indexing         done
```

1. **Extract** -- Pull text from the source. Plain text passes through directly. URLs are validated for SSRF safety (see [Security](#security)) then fetched and converted from HTML to clean text (via trafilatura). PDFs, images, and audio go through their respective parsers (PyMuPDF, pytesseract, Whisper).

2. **Chunk** -- Split extracted text into ~500 token pieces. The chunker is content-aware: markdown splits on headings, code splits on function/class boundaries, transcripts split on speaker turns.

3. **Embed** -- Generate vector embeddings for each chunk using the configured provider. Each chunk is stored as a `fact` memory.

4. **LLM Extract** -- An LLM reads each chunk and extracts structured memories: facts, preferences, episodes, insights, and tasks. It assigns confidence scores and tags. Each extracted memory gets its own embedding.

5. **Graph Build** -- Extracted memories are linked to existing ones through three relationship types:
   - **updates** -- new memory supersedes an old one (marks old as not latest)
   - **extends** -- new memory adds detail to an existing one (both remain active)
   - **derives** -- system-inferred connection based on similarity

### Hybrid search

Search runs two queries in parallel, then fuses scores:

1. **Vector search** -- cosine similarity via pgvector (`1 - (embedding <=> query_vector)`)
2. **BM25 search** -- PostgreSQL full-text ranking via `ts_rank` on auto-maintained `tsvector` columns

Final score: `vector_score * 0.7 + bm25_score * 0.3`. Only memories with `is_latest=TRUE` are returned, so superseded facts are automatically excluded.

### Storage

PostgreSQL with pgvector handles everything in one database:

- `documents` -- raw ingested content and processing status
- `memories` -- extracted knowledge units with vector embeddings
- `memory_relationships` -- graph edges connecting memories
- `containers` -- agent namespace tracking with auto-maintained counts

---

## OpenClaw plugin

The `memory-memorydb/` directory contains a TypeScript plugin that connects [OpenClaw](https://github.com/nicepkg/openclaw) agents to MemoryDB. It gives every agent automatic long-term memory across all channels (Slack, Telegram, Discord, etc.) with no per-channel setup.

> **Directory naming:** The plugin directory is named `memory-memorydb` to match the plugin's entry key in `openclaw.json`. OpenClaw's plugin loader warns when the directory name doesn't match the registered plugin name. If you see a path mismatch warning on gateway restart, make sure the directory name and the key under `plugins.entries` are the same.

### What it does

**Tools** (available to the agent during conversations):
- `memory_search` -- hybrid search with optional container inheritance
- `memory_store` -- save information to long-term memory
- `memory_forget` -- delete memories by ID or search query (GDPR-compliant)
- `memory_profile` -- get a synthesized profile/summary for a container

**Automatic behaviors:**
- **Conversation logging** -- logs all inbound and outbound messages to MemoryDB with channel metadata (`channelId`, `conversationId`, `from`/`to`). Every channel is covered uniformly.
- **Session profile preload** -- fetches the container profile (key facts + recent context) at session start and injects it into the agent's first turn as an `<agent-profile>` block. One-shot per session to avoid redundant API calls.
- **Auto-recall** -- before each agent turn, searches MemoryDB with the user's prompt and injects the top-N results as `<relevant-memories>` context.
- **Auto-capture** -- after each conversation turn, scans user messages with regex heuristics for memorable content (preferences, decisions, entities, explicit "remember that..." requests). Max 3 captures per turn. No LLM calls.
- **Compaction archiving** -- when OpenClaw compacts a long conversation, logs the compaction event so the system knows context was compressed.
- **Session lifecycle** -- logs session end events with message count and duration.

**CLI and commands:**
- `/memory <query>` -- quick search without invoking the AI agent
- `openclaw memorydb status` -- check connection and stats
- `openclaw memorydb search <query>` -- search from the command line
- `openclaw memorydb profile [container]` -- show container profile

### Plugin installation

The plugin requires a running MemoryDB instance and Node.js (for dependency installation).

**Step 1: Install plugin dependencies**

```bash
cd memory-memorydb
npm install
```

This installs `@sinclair/typebox` which is needed at runtime for tool parameter schemas.

**Step 2: Add the plugin to your OpenClaw config**

Point OpenClaw at the plugin directory via `plugins.load.paths`, enable it, and assign it the memory slot. The directory name must match the plugin entry key (`memory-memorydb`):

```json
{
  "plugins": {
    "load": {
      "paths": ["/path/to/memorydb/memory-memorydb"]
    },
    "slots": {
      "memory": "memory-memorydb"
    },
    "entries": {
      "memory-memorydb": {
        "enabled": true,
        "config": {
          "baseUrl": "http://localhost:8080",
          "container": "openclaw.my-agent",
          "autoRecall": true,
          "autoCapture": true,
          "conversationLog": true,
          "sessionProfile": true,
          "inherit": true
        }
      }
    }
  }
}
```

**Step 3: Restart the gateway**

```bash
openclaw gateway restart
```

### Plugin configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | string | `http://localhost:8080` | MemoryDB API URL |
| `container` | string | `default` | Default container/namespace for memories |
| `autoRecall` | boolean | `false` | Inject relevant memories before each agent turn |
| `autoCapture` | boolean | `false` | Auto-detect and store important info from conversations |
| `autoRecallLimit` | number | `5` | Max memories to inject per auto-recall |
| `conversationLog` | boolean | `true` | Log all inbound/outbound messages to MemoryDB |
| `sessionProfile` | boolean | `true` | Preload container profile at session start |
| `inherit` | boolean | `true` | Include parent container results when searching |
| `apiKey` | string | -- | Bearer token for API auth (if MemoryDB is configured with auth) |

### Docker Compose deployment

When running MemoryDB and OpenClaw together via Docker Compose, the plugin connects over the Docker network. Use the service name as the host:

```json
{
  "baseUrl": "http://memorydb:8080",
  "container": "openclaw.my-agent"
}
```

Mount the plugin directory into the OpenClaw container and add it to the load paths. Make sure to run `npm install` in the plugin directory at build time or via an entrypoint so `node_modules/` exists when the plugin loads.

### Container naming convention

Use dot-notation to organize memories hierarchically:

```
openclaw                    # shared system-level knowledge
openclaw.my-agent           # agent-specific memories
openclaw.my-agent.slack     # channel-specific (if needed)
```

With `inherit: true`, searching `openclaw.my-agent` also returns results from `openclaw`. This lets agents share common knowledge while keeping their own memories isolated.

### How auto-capture works

Auto-capture uses rule-based heuristics (no LLM calls) to detect messages worth remembering:

- Explicit requests ("remember that...")
- Preferences ("I prefer...", "I like...", "I hate...")
- Decisions ("we decided...", "going with...", "switching to...")
- Personal facts ("my X is...", "always...", "never...")
- Entities (emails, phone numbers, IP addresses)

Captured content is tagged automatically (`preference`, `decision`, `entity`, `fact`, `explicit`) and stored with `source: auto-capture` metadata.

**Limits:** Max 3 captures per conversation turn. Messages under 10 chars or over 1000 chars are skipped. System boilerplate and XML blocks are filtered out.

### How auto-recall works

Before each agent turn, the user's prompt is searched against stored memories (with `inherit` if configured). The top N results (by hybrid similarity score) are injected as context:

```xml
<relevant-memories>
The following memories from long-term storage may be relevant:
- [87%] User prefers PostgreSQL over MySQL for production (preference)
- [72%] API rate limit is 100 req/min for free tier (fact)
</relevant-memories>
```

The agent sees these memories as additional context and can use them naturally. Prompts shorter than 5 characters are skipped to avoid noise.

### How session profiles work

On `session_start`, the plugin fetches the container profile from MemoryDB (high-confidence facts + recent context) and caches it. On the first `before_agent_start` of that session, the cached profile is injected:

```xml
<agent-profile>
Key facts:
- User prefers TypeScript over JavaScript
- Production environment runs on AWS ECS
Recent context:
- Discussed migration plan for auth service
- Reviewed PR #142 for rate limiting
</agent-profile>
```

The profile is injected once per session (not every turn) to avoid redundant API calls and context bloat.

---

## Updating

### Updating MemoryDB

```bash
cd memorydb
git pull
docker compose up -d --build
```

The `--build` flag rebuilds the MemoryDB container with the latest code. PostgreSQL data persists in the `postgres_data` volume across rebuilds.

If the database schema has changed (check `scripts/init_db.sql` in the diff), you may need to run migrations manually. The init script only runs on first database creation -- it does not auto-migrate existing databases. Check the commit log for migration instructions.

### Updating the OpenClaw plugin

Since the plugin is loaded from the local filesystem via `plugins.load.paths`, updating is:

```bash
cd memorydb
git pull
cd memory-memorydb
npm install    # in case dependencies changed
```

Then restart the OpenClaw gateway to pick up the changes:

```bash
openclaw gateway restart
```

No reinstallation or config changes needed -- the plugin loads from the same path.

---

## Security

### API authentication

Set `API_TOKEN` in `.env` to require a bearer token on all API requests. When set, clients must include `Authorization: Bearer <token>` in every request. When empty, the API is unauthenticated (suitable for localhost-only deployments).

### SSRF protection

The URL extractor validates all user-supplied URLs before fetching. This prevents attackers (or prompt-injected agents) from using the API to reach internal services, cloud metadata endpoints, or localhost.

**What's blocked:**

- Non-HTTP schemes (`ftp://`, `file://`, etc.)
- Cloud metadata hostnames (`metadata.google.internal`)
- Any hostname that resolves to a non-public IP: loopback (`127.x`), RFC 1918 (`10.x`, `172.16-31.x`, `192.168.x`), link-local (`169.254.x`), multicast, reserved, and IPv6 equivalents (`::1`, `fe80::`, etc.)
- Redirect-based bypasses -- every redirect target is re-validated before following, so a public host that 302s to an internal IP is also blocked

Blocked requests raise `SSRFError` (a `ValueError` subclass) with a descriptive message.

### Bind address

Set `API_BIND=127.0.0.1` (the default) to only accept connections from localhost. Use `0.0.0.0` only if the API needs to be reachable over the network, and pair it with `API_TOKEN` for authentication.

---

## Bulk ingestion

The `scripts/ingest_openclaw.py` script bulk-ingests OpenClaw agent files (SOUL.md, MEMORY.md, configs, etc.) into MemoryDB, organized by container:

```bash
python scripts/ingest_openclaw.py --base-url http://localhost:8080
```

Options: `--dry-run` (preview without ingesting), `--delay 0.5` (rate limit between API calls).

---

## License

MIT
