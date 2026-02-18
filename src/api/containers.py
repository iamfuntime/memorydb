from fastapi import APIRouter, HTTPException, Query, Request

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
    storage = _get_storage(request)
    result = await storage.list_containers(prefix, limit)
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
    storage = _get_storage(request)
    count = await storage.delete_container(container_name)
    return {
        "container": container_name,
        "deleted_count": count,
        "success": True,
    }
