import hashlib
import json
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
                json.dumps(metadata or {}),
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
                json.dumps(metadata or {}),
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
                params.append(json.dumps(value))
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
    ) -> Optional[UUID]:
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
                json.dumps(metadata or {}),
            )
            return row["id"] if row else None

    async def get_related_memories(
        self, memory_id: UUID, relationship_type: Optional[str] = None
    ) -> list[dict]:
        conditions = ["r.source_id = $1"]
        params: list = [memory_id]
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
            count = int(result.split()[-1])
            await conn.execute(
                "DELETE FROM containers WHERE name = $1", name
            )
            return count
