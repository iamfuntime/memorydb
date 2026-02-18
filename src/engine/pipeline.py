import asyncio
from typing import Optional
from uuid import UUID

from src.engine.extractors import get_extractor
from src.engine.chunkers import get_chunker
from src.engine.extraction import MemoryExtractor
from src.engine.graph import GraphBuilder
from src.engine.providers.embedding import EmbeddingProvider
from src.engine.providers.llm import LLMProvider
from src.engine.storage import MemoryStorage
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingPipeline:
    """Async pipeline: document -> extract -> chunk -> embed -> LLM extract -> graph."""

    def __init__(
        self,
        storage: MemoryStorage,
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.memory_extractor = (
            MemoryExtractor(llm_provider) if llm_provider else None
        )
        self.graph_builder = GraphBuilder(
            storage, embedding_provider, llm_provider
        )

    async def process_document(self, doc_id: UUID, **kwargs) -> None:
        """Process a document through the full pipeline."""
        try:
            doc = await self.storage.get_document(doc_id)
            if not doc:
                logger.error("pipeline.document_not_found", doc_id=str(doc_id))
                return

            container = doc["container"]
            content_type = doc["content_type"]
            content = doc["raw_content"] or doc["source_url"] or ""

            # Stage 1: Extract text from content
            await self.storage.update_document_status(doc_id, "extracting")
            text = await self._extract(content_type, content, **kwargs)

            if not text.strip():
                await self.storage.update_document_status(
                    doc_id, "failed", "No text extracted"
                )
                return

            # Stage 2: Chunk text
            await self.storage.update_document_status(doc_id, "chunking")
            chunks = self._chunk(content_type, text)

            if not chunks:
                await self.storage.update_document_status(
                    doc_id, "failed", "No chunks produced"
                )
                return

            # Stage 3: Embed chunks + store as raw memories
            await self.storage.update_document_status(doc_id, "embedding")
            chunk_memory_ids = []
            for chunk in chunks:
                try:
                    embedding = await self.embedding_provider.embed(chunk)
                    mem_id = await self.storage.add_memory(
                        container=container,
                        content=chunk,
                        memory_type="fact",
                        document_id=doc_id,
                        embedding=embedding,
                        metadata={"source_type": content_type, "is_chunk": True},
                    )
                    chunk_memory_ids.append(mem_id)
                except Exception as e:
                    logger.warning(
                        "pipeline.embed_chunk_failed",
                        error=str(e),
                        chunk_len=len(chunk),
                    )

            # Stage 4: LLM extraction (if provider available)
            await self.storage.update_document_status(doc_id, "indexing")
            if self.memory_extractor:
                for chunk in chunks:
                    extracted = await self.memory_extractor.extract(chunk)
                    for mem in extracted:
                        try:
                            embedding = await self.embedding_provider.embed(
                                mem["content"]
                            )
                            mem_id = await self.storage.add_memory(
                                container=container,
                                content=mem["content"],
                                memory_type=mem["type"],
                                document_id=doc_id,
                                embedding=embedding,
                                tags=mem.get("tags", []),
                                confidence=mem.get("confidence", 0.8),
                                metadata={
                                    "source_type": content_type,
                                    "extracted": True,
                                },
                            )

                            # Stage 5: Build graph relationships
                            await self.graph_builder.build_relationships(
                                mem_id, container
                            )

                        except Exception as e:
                            logger.warning(
                                "pipeline.extraction_failed",
                                error=str(e),
                                content=mem["content"][:100],
                            )

            await self.storage.update_document_status(doc_id, "done")
            logger.info(
                "pipeline.complete",
                doc_id=str(doc_id),
                chunks=len(chunks),
                extracted=len(chunk_memory_ids),
            )

        except Exception as e:
            logger.error(
                "pipeline.failed", doc_id=str(doc_id), error=str(e)
            )
            await self.storage.update_document_status(
                doc_id, "failed", str(e)
            )

    async def _extract(self, content_type: str, content: str, **kwargs) -> str:
        extractor = get_extractor(content_type)
        return await extractor.extract(content, **kwargs)

    def _chunk(self, content_type: str, text: str) -> list[str]:
        chunker = get_chunker(content_type)
        return chunker.chunk(text)

    def schedule(self, doc_id: UUID, **kwargs) -> asyncio.Task:
        """Schedule document processing as a background task."""
        return asyncio.create_task(self.process_document(doc_id, **kwargs))
