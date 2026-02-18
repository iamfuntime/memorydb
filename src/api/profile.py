from fastapi import APIRouter, HTTPException, Request

from src.engine.profile import ProfileBuilder
from src.models.profile import ProfileResponse

router = APIRouter()


def _get_pool(request: Request):
    storage = request.app.state.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return storage.pool


@router.get("/{container}", response_model=ProfileResponse)
async def get_profile(
    container: str,
    request: Request,
    fact_limit: int = 20,
    recent_limit: int = 10,
):
    """Get a profile for a container."""
    pool = _get_pool(request)
    builder = ProfileBuilder(pool)
    result = await builder.build(
        container,
        fact_limit=fact_limit,
        recent_limit=recent_limit,
    )
    return ProfileResponse(**result)
