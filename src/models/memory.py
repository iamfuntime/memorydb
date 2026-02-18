from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import StrEnum


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    EPISODE = "episode"
    INSIGHT = "insight"
    TASK = "task"


class RelationshipType(StrEnum):
    UPDATES = "updates"
    EXTENDS = "extends"
    DERIVES = "derives"


class MemoryResponse(BaseModel):
    id: UUID
    container: str
    content: str
    memory_type: str
    tags: list[str] = []
    confidence: float = 1.0
    is_latest: bool = True
    created_at: datetime
    updated_at: datetime


class RelatedMemory(BaseModel):
    id: UUID
    content: str
    memory_type: str
    relationship: str
    confidence: float
