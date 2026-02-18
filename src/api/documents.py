from fastapi import APIRouter, HTTPException, Request
from uuid import UUID

import asyncpg

from src.models.document import DocumentCreate, DocumentResponse, DocumentDetail
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_storage(request: Request):
    storage = request.app.state.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return storage


@router.post("", response_model=DocumentResponse, status_code=202)
async def create_document(doc: DocumentCreate, request: Request):
    """Ingest a new document for processing."""
    storage = _get_storage(request)

    if not doc.content or not doc.content.strip():
        raise HTTPException(status_code=400, detail="Content must not be empty")
    if not doc.container or not doc.container.strip():
        raise HTTPException(status_code=400, detail="Container must not be empty")

    source_url = doc.content if doc.content_type == "url" else None
    raw_content = doc.content if doc.content_type != "url" else None

    try:
        doc_id = await storage.add_document(
            container=doc.container,
            content_type=doc.content_type,
            raw_content=raw_content,
            source_url=source_url,
            metadata={**doc.metadata, "tags": doc.tags},
        )
    except asyncpg.PostgresError as e:
        logger.error("documents.create_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create document")

    # Schedule background processing if pipeline is available
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline:
        pipeline.schedule(doc_id)
        logger.info("documents.processing_scheduled", doc_id=str(doc_id))

    document = await storage.get_document(doc_id)
    return DocumentResponse(
        id=document["id"],
        container=document["container"],
        content_type=document["content_type"],
        status=document["status"],
        created_at=document["created_at"],
    )


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: UUID, request: Request):
    """Get document status and details."""
    storage = _get_storage(request)
    try:
        document = await storage.get_document(doc_id)
    except asyncpg.PostgresError as e:
        logger.error("documents.get_failed", doc_id=str(doc_id), error=str(e))
        raise HTTPException(status_code=500, detail="Database error")
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(**document)


@router.delete("/{doc_id}")
async def delete_document(doc_id: UUID, request: Request):
    """Delete a document and its associated memories."""
    storage = _get_storage(request)
    try:
        deleted = await storage.delete_document(doc_id)
    except asyncpg.PostgresError as e:
        logger.error("documents.delete_failed", doc_id=str(doc_id), error=str(e))
        raise HTTPException(status_code=500, detail="Database error")
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"id": str(doc_id), "deleted": True}


@router.post("/list")
async def list_documents(
    request: Request,
    container: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List documents with filtering."""
    if limit < 0 or offset < 0:
        raise HTTPException(status_code=400, detail="limit and offset must be non-negative")
    if limit > 1000:
        raise HTTPException(status_code=400, detail="limit must not exceed 1000")

    storage = _get_storage(request)
    try:
        docs = await storage.list_documents(container, status, limit, offset)
    except asyncpg.PostgresError as e:
        logger.error("documents.list_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")
    return {"documents": docs, "total": len(docs)}
