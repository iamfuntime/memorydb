from fastapi import APIRouter, HTTPException, Request

import asyncpg

from src.models.search import SearchRequest, SearchResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def search_memories(search: SearchRequest, request: Request):
    """Search memories with hybrid ranking (vector + BM25)."""
    search_engine = getattr(request.app.state, "search_engine", None)
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    if not search.query or not search.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        result = await search_engine.search(
            query=search.query,
            container=search.container,
            containers=search.containers,
            inherit=search.inherit,
            limit=search.limit,
            offset=search.offset,
            include_related=search.include_related,
        )
    except asyncpg.PostgresError as e:
        logger.error("search.query_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Search query failed")
    except Exception as e:
        logger.error("search.unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Search failed")

    return result
