from typing import Optional

import asyncpg

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProfileBuilder:
    """Build agent profiles from container memories."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def build(
        self,
        container: str,
        query: Optional[str] = None,
        fact_limit: int = 20,
        recent_limit: int = 10,
    ) -> dict:
        """Build a profile for a container.

        Returns static facts (high-confidence, latest), recent memories
        (last 7 days), and optionally query-relevant memories.
        """
        static_facts = await self._get_static_facts(container, fact_limit)
        recent_context = await self._get_recent(container, recent_limit)

        return {
            "container": container,
            "static_facts": static_facts,
            "recent_context": recent_context,
        }

    async def _get_static_facts(
        self, container: str, limit: int
    ) -> list[str]:
        """Get high-confidence, current facts for a container."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE container = $1
                    AND is_latest = TRUE
                    AND memory_type IN ('fact', 'preference')
                    AND confidence >= 0.7
                ORDER BY confidence DESC, created_at DESC
                LIMIT $2
                """,
                container,
                limit,
            )
            return [r["content"] for r in rows]

    async def _get_recent(
        self, container: str, limit: int
    ) -> list[str]:
        """Get recent memories from the last 7 days."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT content FROM memories
                WHERE container = $1
                    AND is_latest = TRUE
                    AND created_at > NOW() - INTERVAL '7 days'
                ORDER BY created_at DESC
                LIMIT $2
                """,
                container,
                limit,
            )
            return [r["content"] for r in rows]
