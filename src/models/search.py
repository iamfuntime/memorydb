from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime


class SearchRequest(BaseModel):
    query: str
    container: Optional[str] = None
    containers: Optional[list[str]] = None
    inherit: bool = False
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    include_related: bool = False
    filters: Optional[dict] = None


class SearchResult(BaseModel):
    id: UUID
    container: str
    content: str
    memory_type: str
    similarity: float
    tags: list[str] = []
    created_at: datetime
    related: list[dict] = []


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
    limit: int
    offset: int
