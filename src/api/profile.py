from fastapi import APIRouter, HTTPException, Request

import asyncpg

from src.engine.profile import ProfileBuilder
from src.models.profile import ProfileResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    if not container or not container.strip():
        raise HTTPException(status_code=400, detail="Container name must not be empty")
    if fact_limit < 0 or recent_limit < 0:
        raise HTTPException(status_code=400, detail="Limits must be non-negative")

    pool = _get_pool(request)
    builder = ProfileBuilder(pool)
    try:
        result = await builder.build(
            container,
            fact_limit=fact_limit,
            recent_limit=recent_limit,
        )
    except asyncpg.PostgresError as e:
        logger.error("profile.build_failed", container=container, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to build profile")
    return ProfileResponse(**result)
