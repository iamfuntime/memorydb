import json
from typing import Optional
from uuid import UUID

import asyncpg

from src.engine.providers.embedding import EmbeddingProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridSearch:
    """Hybrid search combining vector similarity and BM25 keyword matching."""

    def __init__(self, pool: asyncpg.Pool, embedding_provider: EmbeddingProvider):
        self.pool = pool
        self.embedding_provider = embedding_provider

    async def search(
        self,
        query: str,
        container: Optional[str] = None,
        containers: Optional[list[str]] = None,
        inherit: bool = False,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        offset: int = 0,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        include_related: bool = False,
    ) -> dict:
        # Build target containers list
        target_containers = self._resolve_containers(container, containers, inherit)

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Run vector + BM25 searches
        vector_results = await self._search_vector(
            query_embedding, target_containers, tags, (limit + offset) * 2
        )
        bm25_results = await self._search_bm25(
            query, target_containers, tags, (limit + offset) * 2
        )

        # Combine results
        combined = self._combine_results(
            vector_results, bm25_results, vector_weight, bm25_weight
        )

        # Sort by score and paginate
        sorted_results = sorted(
            combined.values(), key=lambda x: x["similarity"], reverse=True
        )
        paginated = sorted_results[offset : offset + limit]

        # Optionally load related memories
        if include_related:
            for result in paginated:
                result["related"] = await self._get_related(result["id"])

        return {
            "query": query,
            "results": paginated,
            "total": len(combined),
            "limit": limit,
            "offset": offset,
        }

    def _resolve_containers(
        self,
        container: Optional[str],
        containers: Optional[list[str]],
        inherit: bool,
    ) -> list[str]:
        if containers:
            return containers
        if not container:
            return []
        result = [container]
        if inherit and "." in container:
            parts = container.split(".")
            for i in range(len(parts) - 1):
                result.append(".".join(parts[: i + 1]))
        return result

    async def _search_vector(
        self,
        query_embedding: list[float],
        target_containers: list[str],
        tags: Optional[list[str]],
        limit: int,
    ) -> list[dict]:
        conditions = ["is_latest = TRUE", "embedding IS NOT NULL"]
        params: list = [str(query_embedding)]
        idx = 2

        if target_containers:
            container_conditions = []
            for c in target_containers:
                if c.endswith("*"):
                    container_conditions.append(f"container LIKE ${idx}")
                    params.append(c[:-1] + "%")
                else:
                    container_conditions.append(f"container = ${idx}")
                    params.append(c)
                idx += 1
            conditions.append(f"({' OR '.join(container_conditions)})")

        if tags:
            conditions.append(f"tags && ${idx}")
            params.append(tags)
            idx += 1

        where = " AND ".join(conditions)
        params.extend([str(query_embedding), limit])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, container, content, memory_type, tags,
                    confidence, created_at,
                    1 - (embedding <=> $1::vector) as similarity
                FROM memories
                WHERE {where}
                ORDER BY embedding <=> ${idx}::vector
                LIMIT ${idx + 1}
                """,
                *params,
            )
            return [dict(r) for r in rows]

    async def _search_bm25(
        self,
        query: str,
        target_containers: list[str],
        tags: Optional[list[str]],
        limit: int,
    ) -> list[dict]:
        conditions = [
            "is_latest = TRUE",
            "content_tsv @@ plainto_tsquery('english', $1)",
        ]
        params: list = [query]
        idx = 2

        if target_containers:
            container_conditions = []
            for c in target_containers:
                if c.endswith("*"):
                    container_conditions.append(f"container LIKE ${idx}")
                    params.append(c[:-1] + "%")
                else:
                    container_conditions.append(f"container = ${idx}")
                    params.append(c)
                idx += 1
            conditions.append(f"({' OR '.join(container_conditions)})")

        if tags:
            conditions.append(f"tags && ${idx}")
            params.append(tags)
            idx += 1

        where = " AND ".join(conditions)
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, container, content, memory_type, tags,
                    confidence, created_at,
                    ts_rank(content_tsv, plainto_tsquery('english', $1)) as bm25_rank
                FROM memories
                WHERE {where}
                ORDER BY bm25_rank DESC
                LIMIT ${idx}
                """,
                *params,
            )
            return [dict(r) for r in rows]

    def _combine_results(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        vector_weight: float,
        bm25_weight: float,
    ) -> dict:
        combined = {}

        # Index vector results
        for r in vector_results:
            rid = str(r["id"])
            combined[rid] = {
                **r,
                "vector_score": float(r.get("similarity", 0)),
                "bm25_score": 0.0,
                "related": [],
            }

        # Normalize BM25 scores
        max_bm25 = max(
            (float(r.get("bm25_rank", 0)) for r in bm25_results), default=0
        )

        for r in bm25_results:
            rid = str(r["id"])
            normalized_bm25 = (
                float(r.get("bm25_rank", 0)) / max_bm25 if max_bm25 > 0 else 0
            )

            if rid in combined:
                combined[rid]["bm25_score"] = normalized_bm25
            else:
                combined[rid] = {
                    **r,
                    "vector_score": 0.0,
                    "bm25_score": normalized_bm25,
                    "related": [],
                }

        # Calculate hybrid scores
        for rid, result in combined.items():
            result["similarity"] = (
                result["vector_score"] * vector_weight
                + result["bm25_score"] * bm25_weight
            )

        return combined

    async def _get_related(self, memory_id: UUID) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT m.id, m.content, m.memory_type, r.relationship_type
                FROM memory_relationships r
                JOIN memories m ON m.id = r.target_id
                WHERE r.source_id = $1
                LIMIT 5
                """,
                memory_id,
            )
            return [
                {
                    "id": str(r["id"]),
                    "content": r["content"],
                    "memory_type": r["memory_type"],
                    "relationship": r["relationship_type"],
                }
                for r in rows
            ]
