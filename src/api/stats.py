from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _get_pool(request: Request):
    storage = request.app.state.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return storage.pool


@router.get("")
async def get_stats(request: Request):
    """Get system-wide statistics."""
    pool = _get_pool(request)

    async with pool.acquire() as conn:
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        mem_count = await conn.fetchval("SELECT COUNT(*) FROM memories")
        rel_count = await conn.fetchval(
            "SELECT COUNT(*) FROM memory_relationships"
        )

        memories_by_type = await conn.fetch(
            """
            SELECT memory_type, COUNT(*) as count
            FROM memories
            WHERE is_latest = TRUE
            GROUP BY memory_type
            ORDER BY count DESC
            """
        )

        docs_by_status = await conn.fetch(
            """
            SELECT status, COUNT(*) as count
            FROM documents
            GROUP BY status
            ORDER BY count DESC
            """
        )

        embedded_count = await conn.fetchval(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        )

        container_count = await conn.fetchval(
            "SELECT COUNT(*) FROM containers"
        )

    return {
        "documents": {
            "total": doc_count,
            "by_status": {r["status"]: r["count"] for r in docs_by_status},
        },
        "memories": {
            "total": mem_count,
            "by_type": {
                r["memory_type"]: r["count"] for r in memories_by_type
            },
            "embedded": embedded_count,
        },
        "relationships": {"total": rel_count},
        "containers": {"total": container_count},
    }
