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
