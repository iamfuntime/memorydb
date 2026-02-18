from fastapi import APIRouter, HTTPException, Request

from src.models.search import SearchRequest, SearchResponse

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def search_memories(search: SearchRequest, request: Request):
    """Search memories with hybrid ranking (vector + BM25)."""
    search_engine = getattr(request.app.state, "search_engine", None)
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    result = await search_engine.search(
        query=search.query,
        container=search.container,
        containers=search.containers,
        inherit=search.inherit,
        limit=search.limit,
        offset=search.offset,
        include_related=search.include_related,
    )

    return result
