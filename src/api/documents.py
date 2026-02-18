from fastapi import APIRouter, HTTPException, Request
from uuid import UUID

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

    source_url = doc.content if doc.content_type == "url" else None
    raw_content = doc.content if doc.content_type != "url" else None

    doc_id = await storage.add_document(
        container=doc.container,
        content_type=doc.content_type,
        raw_content=raw_content,
        source_url=source_url,
        metadata={**doc.metadata, "tags": doc.tags},
    )

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
    document = await storage.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(**document)


@router.delete("/{doc_id}")
async def delete_document(doc_id: UUID, request: Request):
    """Delete a document and its associated memories."""
    storage = _get_storage(request)
    deleted = await storage.delete_document(doc_id)
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
    storage = _get_storage(request)
    docs = await storage.list_documents(container, status, limit, offset)
    return {"documents": docs, "total": len(docs)}
