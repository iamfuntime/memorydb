from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import StrEnum


class ContentType(StrEnum):
    TEXT = "text"
    URL = "url"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    CSV = "csv"
    JSON_DATA = "json"
    DOCX = "docx"
    XLSX = "xlsx"


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


class DocumentCreate(BaseModel):
    content: str
    container: str
    content_type: str = "text"
    metadata: dict = {}
    tags: list[str] = []


class DocumentResponse(BaseModel):
    id: UUID
    container: str
    content_type: str
    status: str
    created_at: datetime


class DocumentDetail(BaseModel):
    id: UUID
    container: str
    content_type: str
    raw_content: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict = {}
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
