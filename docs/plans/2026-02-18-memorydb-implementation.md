# MemoryDB Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-hosted AI agent memory system with knowledge graph, hybrid search, pluggable providers, and full content type support.

**Architecture:** Monolith FastAPI service with async background processing pipeline. PostgreSQL + pgvector for storage. Documents are ingested, processed (extract, chunk, embed), then LLM-extracted into graph-connected memories.

**Tech Stack:** Python 3.11+, FastAPI, asyncpg, PostgreSQL 16 + pgvector, Pydantic v2, Docker, OpenAI/Anthropic/Ollama (pluggable)

**Design Doc:** `docs/plans/2026-02-18-memorydb-design.md`

---

## Phase 1: Foundation — Docker + DB + Basic CRUD

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `LICENSE`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "memorydb"
version = "0.1.0"
description = "Self-hosted AI agent memory system"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "asyncpg>=0.30.0",
    "pgvector>=0.3.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
    "openai>=1.60.0",
    "anthropic>=0.40.0",
    "httpx>=0.28.0",
    "python-multipart>=0.0.18",
    "structlog>=24.4.0",
]

[project.optional-dependencies]
extractors = [
    "trafilatura>=2.0.0",
    "pymupdf>=1.25.0",
    "pytesseract>=0.3.13",
    "tree-sitter>=0.24.0",
    "python-docx>=1.1.0",
    "openpyxl>=3.1.0",
    "pandas>=2.2.0",
]
local = [
    "sentence-transformers>=3.3.0",
]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.25.0",
    "pytest-cov>=6.0.0",
    "httpx>=0.28.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
asyncpg>=0.30.0
pgvector>=0.3.0
pydantic>=2.10.0
pydantic-settings>=2.7.0
openai>=1.60.0
anthropic>=0.40.0
httpx>=0.28.0
python-multipart>=0.0.18
structlog>=24.4.0
trafilatura>=2.0.0
pymupdf>=1.25.0
pytesseract>=0.3.13
python-docx>=1.1.0
openpyxl>=3.1.0
pandas>=2.2.0
pytest>=8.3.0
pytest-asyncio>=0.25.0
pytest-cov>=6.0.0
```

**Step 3: Create .env.example**

```bash
# PostgreSQL
POSTGRES_DB=memorydb
POSTGRES_USER=memory
POSTGRES_PASSWORD=changeme
POSTGRES_PORT=5432

# Database URL (constructed from above or set directly)
DATABASE_URL=postgresql+asyncpg://memory:changeme@localhost:5432/memorydb

# Embedding Provider: openai, ollama, sentence-transformers
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# LLM Provider: openai, anthropic, ollama
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# API Keys (required for cloud providers)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Ollama (for local providers)
OLLAMA_BASE_URL=http://localhost:11434

# API
API_PORT=8080
LOG_LEVEL=info
```

**Step 4: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.env
*.db
.tmp/
data/
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/
```

**Step 5: Create LICENSE (MIT)**

Standard MIT license text with current year.

**Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore LICENSE
git commit -m "chore: project scaffolding"
```

---

### Task 2: Database Schema

**Files:**
- Create: `scripts/init_db.sql`

**Step 1: Write init_db.sql**

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table (raw ingested content)
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

CREATE INDEX idx_documents_container ON documents(container);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created ON documents(created_at DESC);

-- Memories table (extracted knowledge units / graph nodes)
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    container VARCHAR(255) NOT NULL,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    memory_type VARCHAR(50) NOT NULL DEFAULT 'fact',
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    confidence FLOAT DEFAULT 1.0,
    is_latest BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_memories_container ON memories(container);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_latest ON memories(is_latest) WHERE is_latest = TRUE;
CREATE INDEX idx_memories_created ON memories(created_at DESC);
CREATE INDEX idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX idx_memories_document ON memories(document_id);

-- Full-text search column
ALTER TABLE memories ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX idx_memories_fts ON memories USING GIN(content_tsv);

-- Vector index (ivfflat for cosine similarity)
-- Note: Create after initial data load for best performance
-- For small datasets (<10k), exact search is fine without this index
-- CREATE INDEX idx_memories_embedding ON memories
--     USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Memory relationships table (graph edges)
CREATE TABLE memory_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, relationship_type)
);

CREATE INDEX idx_relationships_source ON memory_relationships(source_id);
CREATE INDEX idx_relationships_target ON memory_relationships(target_id);
CREATE INDEX idx_relationships_type ON memory_relationships(relationship_type);

-- Containers table (agent namespace tracking)
CREATE TABLE containers (
    name VARCHAR(255) PRIMARY KEY,
    description TEXT,
    parent VARCHAR(255) REFERENCES containers(name) ON DELETE SET NULL,
    memory_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trigger to auto-update container memory counts
CREATE OR REPLACE FUNCTION update_container_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO containers (name, memory_count)
            VALUES (NEW.container, 1)
            ON CONFLICT (name)
            DO UPDATE SET memory_count = containers.memory_count + 1;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE containers
            SET memory_count = GREATEST(memory_count - 1, 0)
            WHERE name = OLD.container;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_container_count
AFTER INSERT OR DELETE ON memories
FOR EACH ROW EXECUTE FUNCTION update_container_count();

-- Updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_documents_updated
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_memories_updated
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

**Step 2: Commit**

```bash
git add scripts/init_db.sql
git commit -m "feat: database schema with pgvector, knowledge graph, and FTS"
```

---

### Task 3: Docker Setup

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Step 1: Write Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Step 2: Write docker-compose.yml**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-memorydb}
      POSTGRES_USER: ${POSTGRES_USER:-memory}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD required}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-memory}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  memorydb:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-memory}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-memorydb}
      EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-openai}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-text-embedding-3-small}
      LLM_PROVIDER: ${LLM_PROVIDER:-openai}
      LLM_MODEL: ${LLM_MODEL:-gpt-4o-mini}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
      LOG_LEVEL: ${LOG_LEVEL:-info}
    ports:
      - "${API_PORT:-8080}:8080"
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    restart: unless-stopped

volumes:
  postgres_data:
```

**Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Docker and docker-compose setup"
```

---

### Task 4: Config + Logger + DB Connection

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/utils/__init__.py`
- Create: `src/utils/logger.py`
- Create: `src/utils/db.py`
- Test: `tests/__init__.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
from src.config import Settings


def test_settings_defaults():
    """Settings should have sensible defaults."""
    s = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        openai_api_key="sk-test",
    )
    assert s.embedding_provider == "openai"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.log_level == "info"
    assert s.api_port == 8080


def test_settings_overrides():
    """Settings should accept overrides."""
    s = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        llm_provider="anthropic",
        llm_model="claude-3-haiku-20240307",
        log_level="debug",
        api_port=9090,
    )
    assert s.embedding_provider == "ollama"
    assert s.llm_provider == "anthropic"
    assert s.api_port == 9090
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write implementations**

```python
# src/__init__.py
```

```python
# src/config.py
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str

    # Embedding provider
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # LLM provider
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # API
    api_port: int = 8080
    log_level: str = "info"

    model_config = {"env_file": ".env", "case_sensitive": False}
```

```python
# src/utils/__init__.py
```

```python
# src/utils/logger.py
import structlog
import logging


def setup_logging(level: str = "info") -> None:
    """Configure structured logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
            if level == "debug"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=log_level, format="%(message)s")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger."""
    return structlog.get_logger(name)
```

```python
# src/utils/db.py
import asyncpg
from typing import Optional

_pool: Optional[asyncpg.Pool] = None


async def get_pool(database_url: str) -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        # asyncpg uses postgresql:// not postgresql+asyncpg://
        url = database_url.replace("+asyncpg", "")
        _pool = await asyncpg.create_pool(url, min_size=2, max_size=10)
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
```

```python
# tests/__init__.py
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ tests/
git commit -m "feat: config, logger, and database connection pool"
```

---

### Task 5: FastAPI Skeleton + Health Endpoint

**Files:**
- Create: `src/main.py`
- Create: `src/api/__init__.py`
- Create: `src/api/health.py`
- Test: `tests/test_health.py`
- Test: `tests/conftest.py`

**Step 1: Write the failing test**

```python
# tests/conftest.py
import pytest
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

```python
# tests/test_health.py
import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_health.py -v`
Expected: FAIL

**Step 3: Write implementations**

```python
# src/api/__init__.py
```

```python
# src/api/health.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
    }
```

```python
# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import health

app = FastAPI(
    title="MemoryDB",
    description="Self-hosted AI agent memory system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
```

**Step 4: Run tests**

Run: `pytest tests/test_health.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/main.py src/api/ tests/conftest.py tests/test_health.py
git commit -m "feat: FastAPI skeleton with health endpoint"
```

---

### Task 6: Pydantic Models

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/document.py`
- Create: `src/models/memory.py`
- Create: `src/models/search.py`
- Create: `src/models/profile.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
from src.models.document import DocumentCreate, DocumentResponse
from src.models.memory import MemoryResponse, MemoryType
from src.models.search import SearchRequest, SearchResponse


def test_document_create_text():
    doc = DocumentCreate(content="hello", container="test", content_type="text")
    assert doc.content == "hello"
    assert doc.tags == []


def test_document_create_url():
    doc = DocumentCreate(
        content="https://example.com", container="test", content_type="url"
    )
    assert doc.content_type == "url"


def test_memory_types():
    assert MemoryType.FACT == "fact"
    assert MemoryType.PREFERENCE == "preference"
    assert MemoryType.EPISODE == "episode"


def test_search_request_defaults():
    req = SearchRequest(query="test")
    assert req.limit == 10
    assert req.include_related is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL

**Step 3: Write implementations**

```python
# src/models/__init__.py
```

```python
# src/models/document.py
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import StrEnum


class ContentType(StrEnum):
    TEXT = "text"
    URL = "url"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    CSV = "csv"
    JSON_DATA = "json"
    DOCX = "docx"
    XLSX = "xlsx"


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


class DocumentCreate(BaseModel):
    content: str
    container: str
    content_type: str = "text"
    metadata: dict = {}
    tags: list[str] = []


class DocumentResponse(BaseModel):
    id: UUID
    container: str
    content_type: str
    status: str
    created_at: datetime


class DocumentDetail(BaseModel):
    id: UUID
    container: str
    content_type: str
    raw_content: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict = {}
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
```

```python
# src/models/memory.py
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import StrEnum


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    EPISODE = "episode"
    INSIGHT = "insight"
    TASK = "task"


class RelationshipType(StrEnum):
    UPDATES = "updates"
    EXTENDS = "extends"
    DERIVES = "derives"


class MemoryResponse(BaseModel):
    id: UUID
    container: str
    content: str
    memory_type: str
    tags: list[str] = []
    confidence: float = 1.0
    is_latest: bool = True
    created_at: datetime
    updated_at: datetime


class RelatedMemory(BaseModel):
    id: UUID
    content: str
    memory_type: str
    relationship: str
    confidence: float
```

```python
# src/models/search.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime


class SearchRequest(BaseModel):
    query: str
    container: Optional[str] = None
    containers: Optional[list[str]] = None
    inherit: bool = False
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    include_related: bool = False
    filters: Optional[dict] = None


class SearchResult(BaseModel):
    id: UUID
    container: str
    content: str
    memory_type: str
    similarity: float
    tags: list[str] = []
    created_at: datetime
    related: list[dict] = []


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
    limit: int
    offset: int
```

```python
# src/models/profile.py
from pydantic import BaseModel
from typing import Optional


class ProfileResponse(BaseModel):
    container: str
    static_facts: list[str] = []
    recent_context: list[str] = []
    relevant_memories: list[dict] = []
```

**Step 4: Run tests**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/ tests/test_models.py
git commit -m "feat: Pydantic models for documents, memories, search, profile"
```

---

### Task 7: Storage Layer — Basic CRUD

**Files:**
- Create: `src/engine/__init__.py`
- Create: `src/engine/storage.py`
- Test: `tests/test_storage.py`

**Step 1: Write the failing test**

```python
# tests/test_storage.py
import pytest
import asyncpg
import os

from src.engine.storage import MemoryStorage

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://memory:changeme@localhost:5432/memorydb_test",
)


@pytest.fixture
async def storage():
    """Create storage with test database."""
    pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2)
    # Run schema
    async with pool.acquire() as conn:
        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "init_db.sql"
        )
        with open(schema_path) as f:
            await conn.execute(f.read())
    s = MemoryStorage(pool)
    yield s
    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS memory_relationships CASCADE")
        await conn.execute("DROP TABLE IF EXISTS memories CASCADE")
        await conn.execute("DROP TABLE IF EXISTS documents CASCADE")
        await conn.execute("DROP TABLE IF EXISTS containers CASCADE")
    await pool.close()


@pytest.mark.asyncio
async def test_add_and_get_document(storage):
    doc_id = await storage.add_document(
        container="test",
        content_type="text",
        raw_content="Test content",
        metadata={},
    )
    assert doc_id is not None
    doc = await storage.get_document(doc_id)
    assert doc["raw_content"] == "Test content"
    assert doc["container"] == "test"
    assert doc["status"] == "queued"


@pytest.mark.asyncio
async def test_add_and_get_memory(storage):
    mem_id = await storage.add_memory(
        container="test",
        content="User prefers dark mode",
        memory_type="preference",
        tags=["ui"],
    )
    assert mem_id is not None
    mem = await storage.get_memory(mem_id)
    assert mem["content"] == "User prefers dark mode"
    assert mem["memory_type"] == "preference"


@pytest.mark.asyncio
async def test_delete_memory(storage):
    mem_id = await storage.add_memory(
        container="test", content="temp", memory_type="fact"
    )
    deleted = await storage.delete_memory(mem_id)
    assert deleted is True
    mem = await storage.get_memory(mem_id)
    assert mem is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/engine/__init__.py
```

```python
# src/engine/storage.py
import hashlib
from typing import Optional
from uuid import UUID

import asyncpg


class MemoryStorage:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    # --- Documents ---

    async def add_document(
        self,
        container: str,
        content_type: str,
        raw_content: Optional[str] = None,
        source_url: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> UUID:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (container, content_type, raw_content,
                    source_url, file_path, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                RETURNING id
                """,
                container,
                content_type,
                raw_content,
                source_url,
                file_path,
                __import__("json").dumps(metadata or {}),
            )
            return row["id"]

    async def get_document(self, doc_id: UUID) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", doc_id
            )
            return dict(row) if row else None

    async def update_document_status(
        self, doc_id: UUID, status: str, error_message: Optional[str] = None
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents SET status = $2, error_message = $3
                WHERE id = $1
                """,
                doc_id,
                status,
                error_message,
            )

    async def delete_document(self, doc_id: UUID) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE id = $1", doc_id
            )
            return result == "DELETE 1"

    async def list_documents(
        self,
        container: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        conditions = []
        params = []
        idx = 1

        if container:
            conditions.append(f"container = ${idx}")
            params.append(container)
            idx += 1
        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, container, content_type, status, created_at, updated_at
                FROM documents {where}
                ORDER BY created_at DESC
                LIMIT ${idx} OFFSET ${idx + 1}
                """,
                *params,
            )
            return [dict(r) for r in rows]

    # --- Memories ---

    async def add_memory(
        self,
        container: str,
        content: str,
        memory_type: str = "fact",
        document_id: Optional[UUID] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        confidence: float = 1.0,
    ) -> UUID:
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (container, document_id, content,
                    content_hash, memory_type, embedding, metadata, tags,
                    confidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
                RETURNING id
                """,
                container,
                document_id,
                content,
                content_hash,
                memory_type,
                str(embedding) if embedding else None,
                __import__("json").dumps(metadata or {}),
                tags or [],
                confidence,
            )
            return row["id"]

    async def get_memory(self, memory_id: UUID) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memories WHERE id = $1", memory_id
            )
            return dict(row) if row else None

    async def delete_memory(self, memory_id: UUID) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1", memory_id
            )
            return result == "DELETE 1"

    async def update_memory(
        self, memory_id: UUID, **kwargs
    ) -> Optional[dict]:
        allowed = {"metadata", "tags", "is_latest", "confidence"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return await self.get_memory(memory_id)

        set_clauses = []
        params = []
        idx = 1
        for key, value in updates.items():
            if key == "metadata":
                set_clauses.append(f"metadata = ${idx}::jsonb")
                params.append(__import__("json").dumps(value))
            else:
                set_clauses.append(f"{key} = ${idx}")
                params.append(value)
            idx += 1

        params.append(memory_id)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE memories SET {', '.join(set_clauses)}
                WHERE id = ${idx}
                RETURNING *
                """,
                *params,
            )
            return dict(row) if row else None

    # --- Relationships ---

    async def add_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        relationship_type: str,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> UUID:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_relationships
                    (source_id, target_id, relationship_type, confidence, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (source_id, target_id, relationship_type) DO NOTHING
                RETURNING id
                """,
                source_id,
                target_id,
                relationship_type,
                confidence,
                __import__("json").dumps(metadata or {}),
            )
            return row["id"] if row else None

    async def get_related_memories(
        self, memory_id: UUID, relationship_type: Optional[str] = None
    ) -> list[dict]:
        conditions = ["r.source_id = $1"]
        params = [memory_id]
        idx = 2

        if relationship_type:
            conditions.append(f"r.relationship_type = ${idx}")
            params.append(relationship_type)

        where = " AND ".join(conditions)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT m.*, r.relationship_type, r.confidence as rel_confidence
                FROM memory_relationships r
                JOIN memories m ON m.id = r.target_id
                WHERE {where}
                ORDER BY r.created_at DESC
                """,
                *params,
            )
            return [dict(r) for r in rows]

    # --- Containers ---

    async def list_containers(
        self, prefix: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        if prefix:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM containers
                    WHERE name LIKE $1
                    ORDER BY name LIMIT $2
                    """,
                    prefix + "%",
                    limit,
                )
        else:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM containers ORDER BY name LIMIT $1",
                    limit,
                )
        return [dict(r) for r in rows]

    async def delete_container(self, name: str) -> int:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE container = $1", name
            )
            # Parse "DELETE N"
            count = int(result.split()[-1])
            await conn.execute(
                "DELETE FROM containers WHERE name = $1", name
            )
            return count
```

**Step 4: Run tests**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (requires running test Postgres)

Note: For CI/local testing without Postgres, these tests will be skipped. Mark with `@pytest.mark.integration`.

**Step 5: Commit**

```bash
git add src/engine/ tests/test_storage.py
git commit -m "feat: storage layer with document, memory, relationship, and container CRUD"
```

---

### Task 8: Document + Memory API Endpoints

**Files:**
- Create: `src/api/documents.py`
- Create: `src/api/memories.py`
- Create: `src/api/containers.py`
- Modify: `src/main.py` (register routers)
- Test: `tests/test_api_documents.py`

**Step 1: Write the failing test**

```python
# tests/test_api_documents.py
import pytest


@pytest.mark.asyncio
async def test_create_document_text(client):
    response = await client.post(
        "/v1/documents",
        json={
            "content": "Test memory content",
            "container": "test-agent",
            "content_type": "text",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "id" in data
    assert data["container"] == "test-agent"
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_create_document_missing_content(client):
    response = await client.post(
        "/v1/documents",
        json={"container": "test"},
    )
    assert response.status_code == 422
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_documents.py -v`
Expected: FAIL

**Step 3: Write implementations**

```python
# src/api/documents.py
from fastapi import APIRouter, HTTPException
from uuid import UUID
from src.models.document import DocumentCreate, DocumentResponse, DocumentDetail

router = APIRouter()

# Note: Storage will be injected via FastAPI dependency injection
# For now, these are stubs that will be wired up when storage is connected

@router.post("", response_model=DocumentResponse, status_code=202)
async def create_document(doc: DocumentCreate):
    """Ingest a new document for processing."""
    # Stub — will be wired to storage + pipeline in integration task
    raise HTTPException(status_code=501, detail="Not yet wired to storage")


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: UUID):
    """Get document status and details."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")


@router.delete("/{doc_id}")
async def delete_document(doc_id: UUID):
    """Delete a document and its associated memories."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")


@router.post("/list")
async def list_documents(
    container: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List documents with filtering."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")
```

```python
# src/api/memories.py
from fastapi import APIRouter, HTTPException
from uuid import UUID

router = APIRouter()


@router.get("/{memory_id}")
async def get_memory(memory_id: UUID):
    """Get a specific memory."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")


@router.delete("/{memory_id}")
async def delete_memory(memory_id: UUID):
    """Delete a memory."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")


@router.get("/{memory_id}/related")
async def get_related_memories(memory_id: UUID, relationship_type: str | None = None):
    """Get memories related to this one via graph edges."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")


@router.patch("/{memory_id}")
async def update_memory(memory_id: UUID, updates: dict):
    """Update memory metadata/tags."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")
```

```python
# src/api/containers.py
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("")
async def list_containers(prefix: str | None = None, limit: int = 100):
    """List all containers."""
    raise HTTPException(status_code=501, detail="Not yet wired to storage")


@router.delete("/{container_name}")
async def delete_container(container_name: str, confirm: bool = Query(False)):
    """Delete all memories in a container."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Must confirm with confirm=true")
    raise HTTPException(status_code=501, detail="Not yet wired to storage")
```

Update `src/main.py` to register the new routers:

```python
# src/main.py — add imports and include_router calls
from src.api import health, documents, memories, containers

# ... existing app setup ...

app.include_router(health.router)
app.include_router(documents.router, prefix="/v1/documents", tags=["documents"])
app.include_router(memories.router, prefix="/v1/memories", tags=["memories"])
app.include_router(containers.router, prefix="/v1/containers", tags=["containers"])
```

**Step 4: Run tests**

Run: `pytest tests/test_health.py -v` (health still passes)
Note: Document API tests need storage wiring — tested in integration task.

**Step 5: Commit**

```bash
git add src/api/ src/main.py tests/
git commit -m "feat: API endpoint stubs for documents, memories, containers"
```

---

### Task 9: Wire Storage to API (Integration)

**Files:**
- Modify: `src/main.py` (startup/shutdown, dependency injection)
- Modify: `src/api/documents.py` (wire to storage)
- Modify: `src/api/memories.py` (wire to storage)
- Modify: `src/api/containers.py` (wire to storage)

**Step 1: Add startup/shutdown lifecycle to main.py**

Wire `get_pool` / `close_pool` into FastAPI lifespan. Create a `MemoryStorage` instance as app state. Use `Depends` to inject storage into endpoints.

**Step 2: Replace all `raise HTTPException(501)` stubs with actual storage calls**

Each endpoint calls `request.app.state.storage.<method>()`.

**Step 3: Run full test suite against running Docker Postgres**

Run: `docker-compose up -d postgres && pytest tests/ -v`
Expected: All tests PASS

**Step 4: Test manually with curl**

```bash
# Store a document
curl -X POST http://localhost:8080/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "container": "test", "content_type": "text"}'

# Check health
curl http://localhost:8080/health
```

**Step 5: Commit**

```bash
git add src/
git commit -m "feat: wire storage layer to API endpoints"
```

---

## Phase 2: Pluggable Providers + Embeddings

### Task 10: Embedding Provider Protocol + Factory

**Files:**
- Create: `src/engine/providers/__init__.py`
- Create: `src/engine/providers/embedding.py`
- Test: `tests/test_embedding_provider.py`

**Step 1: Write the failing test**

```python
# tests/test_embedding_provider.py
import pytest
from src.engine.providers.embedding import get_embedding_provider


def test_get_openai_provider():
    provider = get_embedding_provider("openai", model="text-embedding-3-small")
    assert provider.dimensions() == 1536


def test_get_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_provider("nonexistent")
```

**Step 2: Write implementation**

Define `EmbeddingProvider` protocol with `embed(text) -> list[float]` and `dimensions() -> int`. Implement factory function.

**Step 3: Commit after tests pass**

---

### Task 11: OpenAI Embedding Provider

**Files:**
- Create: `src/engine/providers/openai.py`
- Test: `tests/test_openai_provider.py`

Implement `OpenAIEmbedding` class using the `openai` SDK. Test with mock.

---

### Task 12: Ollama Embedding Provider

**Files:**
- Create: `src/engine/providers/ollama.py`
- Test: `tests/test_ollama_provider.py`

Implement `OllamaEmbedding` class using `httpx` calls to Ollama API. Configurable model and dimensions.

---

### Task 13: Embed on Ingest + Vector Search

**Files:**
- Modify: `src/engine/storage.py` (add `search_vector` method)
- Create: `src/api/search.py`
- Test: `tests/test_search.py`

**Key implementation:** When a memory is added, call `provider.embed(content)` and store the vector. Add `search_vector()` to storage that does `1 - (embedding <=> query_vector)` cosine similarity via pgvector.

Wire `/v1/search` endpoint.

**Commit after tests pass.**

---

## Phase 3: Hybrid Search + BM25

### Task 14: BM25 Search via PostgreSQL Full-Text Search

**Files:**
- Modify: `src/engine/storage.py` (add `search_bm25` method)
- Test: `tests/test_bm25.py`

Use `ts_rank(content_tsv, plainto_tsquery('english', $1))` for BM25-style ranking.

---

### Task 15: Hybrid Search Engine

**Files:**
- Create: `src/engine/search.py`
- Test: `tests/test_hybrid_search.py`

Combine vector + BM25 results. Configurable weights (default 0.7 vector / 0.3 BM25). Normalize BM25 scores to 0-1 range. Support container filtering with wildcards and inheritance.

---

### Task 16: Wire Search API

**Files:**
- Modify: `src/api/search.py`
- Test: `tests/test_api_search.py`

Accept `SearchRequest`, return `SearchResponse` with hybrid scores.

**Commit after tests pass.**

---

## Phase 4: Content Extractors + Smart Chunking

### Task 17: Extractor Protocol + Registry

**Files:**
- Create: `src/engine/extractors/__init__.py`
- Create: `src/engine/extractors/base.py`
- Test: `tests/test_extractors.py`

Define `ContentExtractor` protocol: `async def extract(content: str, **kwargs) -> str`. Registry maps `content_type -> Extractor`.

---

### Task 18: Text Extractor (pass-through)

**Files:**
- Create: `src/engine/extractors/text.py`

---

### Task 19: URL Extractor

**Files:**
- Create: `src/engine/extractors/url.py`

Use `trafilatura` to fetch URL and extract clean text.

---

### Task 20: PDF Extractor

**Files:**
- Create: `src/engine/extractors/pdf.py`

Use `pymupdf` for text extraction with `pytesseract` OCR fallback.

---

### Task 21: Image, Audio, Code, Structured Data Extractors

**Files:**
- Create: `src/engine/extractors/image.py`
- Create: `src/engine/extractors/audio.py`
- Create: `src/engine/extractors/code.py`
- Create: `src/engine/extractors/structured.py`

---

### Task 22: Chunker Protocol + Text Chunker

**Files:**
- Create: `src/engine/chunkers/__init__.py`
- Create: `src/engine/chunkers/base.py`
- Create: `src/engine/chunkers/text.py`
- Test: `tests/test_chunkers.py`

Split by headings, then paragraphs. Target ~500 tokens per chunk.

---

### Task 23: Code Chunker + Transcript Chunker

**Files:**
- Create: `src/engine/chunkers/code.py`
- Create: `src/engine/chunkers/transcript.py`

---

### Task 24: Processing Pipeline Orchestrator

**Files:**
- Create: `src/engine/pipeline.py`
- Test: `tests/test_pipeline.py`

Async pipeline: `document -> extract -> chunk -> embed -> store memories`. Updates document status at each stage. Runs as asyncio background task.

Wire into document creation endpoint so `POST /v1/documents` queues the pipeline.

**Commit after tests pass.**

---

## Phase 5: Knowledge Graph + LLM Extraction

### Task 25: LLM Provider Protocol + Factory

**Files:**
- Create: `src/engine/providers/llm.py`
- Test: `tests/test_llm_provider.py`

Define `LLMProvider` protocol: `async def complete(prompt, system) -> str`. Factory selects by config.

---

### Task 26: OpenAI, Anthropic, Ollama LLM Providers

**Files:**
- Modify: `src/engine/providers/openai.py` (add LLM class)
- Create: `src/engine/providers/anthropic.py`
- Modify: `src/engine/providers/ollama.py` (add LLM class)

---

### Task 27: Memory Extraction Prompt + Classifier

**Files:**
- Create: `src/engine/extraction.py`
- Test: `tests/test_extraction.py`

LLM prompt that takes chunks and outputs structured JSON: `{memories: [{content, type, confidence, tags}]}`. Parse response, validate, store.

---

### Task 28: Graph Relationship Builder

**Files:**
- Create: `src/engine/graph.py`
- Test: `tests/test_graph.py`

After extracting memories: search for semantically similar existing memories, use LLM to determine relationships (updates/extends/derives), store edges, handle contradictions (set `is_latest=false` on superseded memories).

---

### Task 29: Wire LLM Extraction into Pipeline

**Files:**
- Modify: `src/engine/pipeline.py`

After chunking + embedding: run LLM extraction, build graph relationships.

**Commit after tests pass.**

---

## Phase 6: Profile Builder + Container Management

### Task 30: Profile Builder

**Files:**
- Create: `src/engine/profile.py`
- Create: `src/api/profile.py`
- Modify: `src/main.py` (register profile router)
- Test: `tests/test_profile.py`

Aggregate: most common facts (by type), recent memories (last 7 days), optionally search-contextualized results. Return structured profile.

---

### Task 31: Container Management + Stats

**Files:**
- Modify: `src/api/containers.py` (implement archive endpoint)
- Create: `src/api/stats.py`
- Modify: `src/main.py` (register stats router)

Stats: total documents, total memories, memories by type, embedding coverage, containers with counts.

**Commit after tests pass.**

---

## Phase 7: Testing + Polish + Documentation

### Task 32: Integration Test Suite

**Files:**
- Create: `tests/integration/test_full_workflow.py`
- Create: `scripts/test_api.sh`

End-to-end: ingest document -> wait for processing -> search -> verify graph relationships -> check profile.

---

### Task 33: Error Handling + Input Validation Audit

Review all endpoints for: missing error handling, invalid input edge cases, proper HTTP status codes, meaningful error messages.

---

### Task 34: README + API Docs

**Files:**
- Create: `README.md`

Quick-start guide, API reference, configuration reference, examples.

**Final commit: tag v0.1.0**

```bash
git tag -a v0.1.0 -m "Initial release: MemoryDB v0.1.0"
```
