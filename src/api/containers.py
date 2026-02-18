from fastapi import APIRouter, HTTPException, Query, Request

import asyncpg

from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_storage(request: Request):
    storage = request.app.state.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return storage


@router.get("")
async def list_containers(
    request: Request,
    prefix: str | None = None,
    limit: int = 100,
):
    """List all containers."""
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be non-negative")
    if limit > 1000:
        raise HTTPException(status_code=400, detail="limit must not exceed 1000")

    storage = _get_storage(request)
    try:
        result = await storage.list_containers(prefix, limit)
    except asyncpg.PostgresError as e:
        logger.error("containers.list_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")
    return {"containers": result, "total": len(result)}


@router.delete("/{container_name}")
async def delete_container(
    container_name: str,
    request: Request,
    confirm: bool = Query(False),
):
    """Delete all memories in a container."""
    if not confirm:
        raise HTTPException(
            status_code=400, detail="Must confirm deletion with confirm=true"
        )
    if not container_name or not container_name.strip():
        raise HTTPException(status_code=400, detail="Container name must not be empty")

    storage = _get_storage(request)
    try:
        count = await storage.delete_container(container_name)
    except asyncpg.PostgresError as e:
        logger.error("containers.delete_failed", container=container_name, error=str(e))
        raise HTTPException(status_code=500, detail="Database error")
    return {
        "container": container_name,
        "deleted_count": count,
        "success": True,
    }
