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
