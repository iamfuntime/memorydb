from fastapi import APIRouter, HTTPException, Request
from uuid import UUID

router = APIRouter()


def _get_storage(request: Request):
    storage = request.app.state.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return storage


@router.get("/{memory_id}")
async def get_memory(memory_id: UUID, request: Request):
    """Get a specific memory."""
    storage = _get_storage(request)
    memory = await storage.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    # Remove embedding from response (too large)
    result = dict(memory)
    result.pop("embedding", None)
    result.pop("content_tsv", None)
    return result


@router.delete("/{memory_id}")
async def delete_memory(memory_id: UUID, request: Request):
    """Delete a memory."""
    storage = _get_storage(request)
    deleted = await storage.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"id": str(memory_id), "deleted": True}


@router.get("/{memory_id}/related")
async def get_related_memories(
    memory_id: UUID,
    request: Request,
    relationship_type: str | None = None,
):
    """Get memories related to this one via graph edges."""
    storage = _get_storage(request)
    # Verify memory exists
    memory = await storage.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    related = await storage.get_related_memories(memory_id, relationship_type)
    results = []
    for r in related:
        item = dict(r)
        item.pop("embedding", None)
        item.pop("content_tsv", None)
        results.append(item)
    return {"memory_id": str(memory_id), "related": results}


@router.patch("/{memory_id}")
async def update_memory(memory_id: UUID, updates: dict, request: Request):
    """Update memory metadata/tags."""
    storage = _get_storage(request)
    result = await storage.update_memory(memory_id, **updates)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    result.pop("embedding", None)
    result.pop("content_tsv", None)
    return result
