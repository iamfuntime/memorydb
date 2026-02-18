# MemoryDB Design Document

**Date**: 2026-02-18
**Status**: Approved
**Goal**: Self-hosted, open-source persistent memory system for AI agents with knowledge graph, hybrid search, and multi-agent support.

---

## Overview

MemoryDB is a self-hosted alternative to Supermemory. It provides AI agents with persistent, intelligent memory via a REST API backed by PostgreSQL + pgvector. Content is ingested, processed through a pipeline (extraction, chunking, embedding, LLM-powered memory extraction), and stored as a knowledge graph where memories connect through update/extend/derive relationships.

### Key Requirements

- Docker container deployment (one-command setup)
- REST API (<50ms for search queries)
- Knowledge graph memory model (not flat storage)
- Hybrid search (vector + BM25 + graph traversal)
- Multi-agent support via container namespacing
- Pluggable embedding providers (OpenAI, Ollama, sentence-transformers)
- Pluggable LLM providers (OpenAI, Anthropic, Ollama)
- Full content type support (text, URLs, PDFs, images, audio, code, structured data)
- Auto-extraction of memories from ingested content via LLM
- Generic/configurable (no hardcoded agent names)
- Open source (MIT license)

### Target Users

- AI agent builders (OpenClaw, LangChain, custom frameworks)
- Self-hosters (privacy-focused, no SaaS dependency)
- Developers building AI applications that need persistent memory

---

## Architecture

```
+---------------------------------------------+
|         AI Agent / Client                    |
|  (OpenClaw, LangChain, Claude Code, etc.)   |
+--------------------+------------------------+
                     | HTTP REST API
                     | localhost:8080
                     v
+---------------------------------------------+
|   MemoryDB Docker Container                  |
|                                              |
|   +--------------------------------------+   |
|   |   FastAPI Server (async)             |   |
|   |   - Memory CRUD                      |   |
|   |   - Hybrid Search                    |   |
|   |   - Profile Builder                  |   |
|   |   - Container Management             |   |
|   |   - Health/Status                    |   |
|   +-----------------+--------------------+   |
|                     |                        |
|   +-----------------v--------------------+   |
|   |   Processing Pipeline (async bg)     |   |
|   |   - Content Extraction               |   |
|   |   - Smart Chunking                   |   |
|   |   - Embedding Generation (pluggable) |   |
|   |   - LLM Extraction (pluggable)       |   |
|   |   - Graph Relationship Builder       |   |
|   +-----------------+--------------------+   |
|                     |                        |
|   +-----------------v--------------------+   |
|   |   Storage Layer                      |   |
|   |   - PostgreSQL + pgvector            |   |
|   |   - Knowledge Graph (memories +      |   |
|   |     relationships tables)            |   |
|   +--------------------------------------+   |
+---------------------------------------------+
```

**Key principles:**
- API returns `201` for simple text, `202 Accepted` for content requiring processing
- All heavy processing happens in async background tasks
- Pluggable providers via Protocol-based adapter pattern
- Container namespacing for multi-agent isolation

---

## Data Model

### `documents` -- Raw ingested content

```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  container VARCHAR(255) NOT NULL,
  content_type VARCHAR(50) NOT NULL,
  raw_content TEXT,
  source_url TEXT,
  file_path TEXT,
  metadata JSONB DEFAULT '{}',
  status VARCHAR(20) DEFAULT 'queued',
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### `memories` -- Extracted knowledge units (graph nodes)

```sql
CREATE TABLE memories (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  container VARCHAR(255) NOT NULL,
  document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
  content TEXT NOT NULL,
  memory_type VARCHAR(50) NOT NULL,
  embedding vector(1536),
  metadata JSONB DEFAULT '{}',
  tags TEXT[] DEFAULT '{}',
  confidence FLOAT DEFAULT 1.0,
  is_latest BOOLEAN DEFAULT TRUE,
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### `memory_relationships` -- Graph edges

```sql
CREATE TABLE memory_relationships (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id UUID REFERENCES memories(id) ON DELETE CASCADE,
  target_id UUID REFERENCES memories(id) ON DELETE CASCADE,
  relationship_type VARCHAR(20) NOT NULL,
  confidence FLOAT DEFAULT 1.0,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(source_id, target_id, relationship_type)
);
```

### `containers` -- Agent namespace tracking

```sql
CREATE TABLE containers (
  name VARCHAR(255) PRIMARY KEY,
  description TEXT,
  parent VARCHAR(255) REFERENCES containers(name),
  memory_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Relationship Types

- **updates**: New memory supersedes old one (sets old `is_latest=false`). Handles contradiction resolution.
- **extends**: New memory adds detail without replacing. Both remain valid and searchable.
- **derives**: System-inferred connection from patterns across memories.

### Memory Types

- **fact**: Objective information ("Team uses PostgreSQL")
- **preference**: User/agent preferences ("Prefers dark mode")
- **episode**: Events that happened ("Meeting on Feb 18")
- **insight**: Learned patterns ("User asks about Docker frequently")
- **task**: Action items ("Need to review PR #42")

### Indexes

- `ivfflat` on `memories.embedding` for vector similarity
- `GIN` on `memories.tags` for tag filtering
- `GIN` on full-text search column (content_tsv) for BM25
- B-tree on `container`, `memory_type`, `is_latest`, `created_at`
- B-tree on `memory_relationships(source_id)` and `memory_relationships(target_id)`

---

## REST API

### Documents (Ingest)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/documents` | Ingest content (text, URL, or base64 file) |
| `POST` | `/v1/documents/upload` | File upload (multipart form) |
| `GET` | `/v1/documents/{id}` | Get document status/details |
| `DELETE` | `/v1/documents/{id}` | Delete document + associated memories |
| `POST` | `/v1/documents/list` | List documents with filtering |

### Memories (Knowledge Graph)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/memories/{id}` | Get a specific memory |
| `DELETE` | `/v1/memories/{id}` | Delete a memory |
| `GET` | `/v1/memories/{id}/related` | Get related memories (graph traversal) |
| `PATCH` | `/v1/memories/{id}` | Update metadata/tags |

### Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/search` | Hybrid search (vector + BM25 + graph) |

### Profile

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/profile` | Get auto-generated user/agent profile |

### Containers

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/containers` | List containers |
| `DELETE` | `/v1/containers/{name}` | Delete container + all memories |
| `POST` | `/v1/containers/{name}/archive` | Archive to parent container |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/stats` | Memory/document counts, embedding coverage |

### Request/Response Examples

**Ingest text:**
```json
POST /v1/documents
{
  "content": "Had a great meeting with the team. We decided to use PostgreSQL.",
  "container": "my-agent",
  "content_type": "text",
  "metadata": {"source": "conversation"},
  "tags": ["meetings", "decisions"]
}
--> 202 Accepted
{
  "id": "550e8400-...",
  "status": "queued",
  "container": "my-agent"
}
```

**Search:**
```json
POST /v1/search
{
  "query": "database decision",
  "container": "my-agent",
  "limit": 10,
  "include_related": true,
  "filters": {
    "memory_type": ["fact", "preference"],
    "tags": ["decisions"]
  }
}
--> 200 OK
{
  "query": "database decision",
  "results": [
    {
      "id": "...",
      "content": "Team decided to use PostgreSQL for the database",
      "memory_type": "fact",
      "similarity": 0.94,
      "related": [
        {"id": "...", "content": "Deploy on Docker", "relationship": "extends"}
      ]
    }
  ],
  "total": 3
}
```

**Profile:**
```json
GET /v1/profile?container=my-agent&query=infrastructure
--> 200 OK
{
  "container": "my-agent",
  "static_facts": ["Uses PostgreSQL", "Deploys on Docker"],
  "recent_context": ["Working on MemoryDB implementation"],
  "relevant_memories": [...]
}
```

---

## Processing Pipeline

### Stages

```
Document --> Extract --> Chunk --> Embed --> Extract Memories --> Build Graph --> Done
  queued   extracting  chunking  embedding     indexing                        done
```

### Content Extractors

| Type | Extractor | Library |
|------|-----------|---------|
| `text` | Pass-through | -- |
| `url` | HTML to Markdown | `trafilatura` or `readability-lxml` |
| `pdf` | Text extraction + OCR fallback | `pymupdf` + `pytesseract` |
| `image` | OCR + optional vision | `pytesseract` + LLM vision API |
| `audio` | Speech-to-text | `openai-whisper` or Groq Whisper API |
| `code` | AST-aware parsing | `tree-sitter` |
| `csv/json` | Structured to text | `pandas` |
| `docx/xlsx` | Document parsing | `python-docx`, `openpyxl` |

### Smart Chunking

- **Text/Markdown**: Split by heading hierarchy, then paragraph. ~500 tokens per chunk.
- **Code**: AST-aware -- functions and classes stay intact.
- **URLs/HTML**: Split by article structure (headers, sections).
- **PDFs**: Split by page + semantic sections.
- **Transcripts**: Split by speaker turns or time windows.

### LLM Memory Extraction

From each set of chunks, the LLM extracts:
1. Individual memories (facts, preferences, episodes, insights, tasks)
2. Memory type classification
3. Confidence scores
4. Relationships to existing memories (updates, extends, derives)
5. Contradiction detection (marks old memories as `is_latest=false`)
6. Expiration detection for temporal facts

---

## Pluggable Provider System

### Embedding Providers

```python
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    def dimensions(self) -> int: ...
```

Implementations: OpenAI (`text-embedding-3-small`), Ollama (`nomic-embed-text`), sentence-transformers.

### LLM Providers

```python
class LLMProvider(Protocol):
    async def complete(self, prompt: str, system: str) -> str: ...
```

Implementations: OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-haiku`), Ollama (any model).

### Configuration

```bash
EMBEDDING_PROVIDER=openai          # or ollama, sentence-transformers
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=openai                # or anthropic, ollama
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Project Structure

```
memorydb/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   ├── memories.py
│   │   ├── search.py
│   │   ├── profile.py
│   │   ├── containers.py
│   │   └── health.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── extractors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── text.py
│   │   │   ├── url.py
│   │   │   ├── pdf.py
│   │   │   ├── image.py
│   │   │   ├── audio.py
│   │   │   ├── code.py
│   │   │   └── structured.py
│   │   ├── chunkers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── text.py
│   │   │   ├── code.py
│   │   │   └── transcript.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py
│   │   │   ├── llm.py
│   │   │   ├── openai.py
│   │   │   ├── anthropic.py
│   │   │   └── ollama.py
│   │   ├── search.py
│   │   ├── graph.py
│   │   ├── profile.py
│   │   └── storage.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── memory.py
│   │   ├── search.py
│   │   └── profile.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── db.py
├── scripts/
│   ├── init_db.sql
│   └── test_api.sh
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_documents.py
│   ├── test_memories.py
│   ├── test_search.py
│   ├── test_graph.py
│   ├── test_extractors.py
│   └── test_pipeline.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Docker Deployment

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: memorydb
      POSTGRES_USER: memory
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U memory"]

  memorydb:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://memory:${POSTGRES_PASSWORD}@postgres:5432/memorydb
      EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-openai}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-text-embedding-3-small}
      LLM_PROVIDER: ${LLM_PROVIDER:-openai}
      LLM_MODEL: ${LLM_MODEL:-gpt-4o-mini}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
    ports:
      - "${API_PORT:-8080}:8080"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

### Technical Choices

- **asyncpg**: Native async PostgreSQL driver (faster than psycopg2)
- **Pydantic v2**: Fast validation, automatic OpenAPI docs
- **Structured logging**: JSON for production, human-readable for dev
- **Connection pooling**: Via asyncpg built-in pool

---

## Implementation Phases

### Phase 1: Foundation -- Docker + DB + Basic CRUD

- Project scaffolding (pyproject.toml, requirements.txt, Dockerfile, docker-compose.yml)
- PostgreSQL schema (init_db.sql)
- FastAPI skeleton with health endpoint
- Config system (Pydantic settings)
- Database connection pool (asyncpg)
- Basic document ingest (text only)
- Basic memory CRUD (manual storage)
- **Deliverable**: `docker-compose up` works, store/retrieve text memories

### Phase 2: Pluggable Providers + Embeddings

- Embedding provider protocol + factory
- OpenAI embedding implementation
- Ollama embedding implementation
- Sentence-transformer embedding implementation
- Embed on ingest
- Vector similarity search endpoint
- **Deliverable**: Store memories with embeddings, search by similarity

### Phase 3: Hybrid Search + BM25

- Full-text search column (tsvector)
- BM25 scoring via PostgreSQL ts_rank
- Hybrid search combining vector + BM25
- Container filtering (exact, wildcard, inheritance)
- Tag filtering, pagination
- **Deliverable**: Production-quality hybrid search

### Phase 4: Content Extractors + Smart Chunking

- Extractor protocol + registry
- All content type extractors (text, URL, PDF, image, audio, code, structured)
- Content-type-aware chunking strategies
- Async processing pipeline
- **Deliverable**: Ingest any content type, get searchable chunks

### Phase 5: Knowledge Graph + LLM Extraction

- LLM provider protocol + implementations
- Memory extraction prompt
- Memory type classification
- Relationship detection (updates, extends, derives)
- Contradiction detection
- Auto-forgetting
- Graph traversal
- **Deliverable**: Auto-extracted, graph-connected memories

### Phase 6: Profile Builder + Container Management

- Profile builder (static facts + recent context)
- Container CRUD (list, delete, archive)
- Container hierarchy
- Stats endpoint
- **Deliverable**: Full profile and container management API

### Phase 7: Testing + Polish + Documentation

- Unit tests for all engine components
- Integration tests for API endpoints
- Test fixtures with mock providers
- Error handling and input validation audit
- README, API reference docs
- **Deliverable**: Production-ready v1.0.0

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Monolith FastAPI | Simple deployment, fast to build, async bg tasks for heavy work |
| Memory model | Knowledge graph | Supermemory-style relationships, contradiction resolution, temporal awareness |
| Database | PostgreSQL + pgvector | Best open-source vector DB, full-text search built in |
| Async driver | asyncpg | Native async, faster than psycopg2 |
| Embeddings | Pluggable (OpenAI/Ollama/ST) | Flexibility: cloud for quality, local for privacy |
| LLM | Pluggable (OpenAI/Anthropic/Ollama) | Same flexibility pattern |
| API | REST only | Simpler, faster, wider compatibility than MCP |
| Content ingest | Full type support | Match supermemory: text, URL, PDF, image, audio, code, structured |
| Memory extraction | Auto via LLM | Intelligent extraction on every ingest |
| Container naming | Generic dot-notation | agent.subagent pattern, no hardcoded names |
