import json
from typing import Optional
from uuid import UUID

from src.engine.providers.embedding import EmbeddingProvider
from src.engine.providers.llm import LLMProvider
from src.engine.storage import MemoryStorage
from src.utils.logger import get_logger

logger = get_logger(__name__)

RELATIONSHIP_SYSTEM_PROMPT = """You are a memory relationship analyzer. Given a new memory and existing similar memories, determine if relationships exist.

Relationship types:
- updates: New memory supersedes/contradicts the old one (e.g., "Works at Google" -> "Now works at Stripe")
- extends: New memory adds detail to the old one without replacing it (both remain valid)
- derives: New memory is logically connected to the old one (inferred relationship)

Respond ONLY with valid JSON:
{
  "relationships": [
    {
      "existing_memory_id": "uuid",
      "type": "updates|extends|derives",
      "confidence": 0.0-1.0,
      "reason": "brief explanation"
    }
  ]
}

If no relationships exist, return: {"relationships": []}
Be conservative - only identify clear relationships."""


class GraphBuilder:
    """Build knowledge graph relationships between memories."""

    def __init__(
        self,
        storage: MemoryStorage,
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.llm = llm_provider

    async def build_relationships(
        self,
        memory_id: UUID,
        container: str,
    ) -> list[dict]:
        """Find and create relationships for a new memory."""
        memory = await self.storage.get_memory(memory_id)
        if not memory:
            return []

        # Find similar existing memories via vector search
        similar = await self._find_similar(
            memory["content"], container, exclude_id=memory_id
        )

        if not similar:
            return []

        if self.llm is None:
            # Without LLM, create "derives" relationships for high-similarity matches
            relationships = []
            for s in similar:
                if s["similarity"] > 0.85:
                    rel_id = await self.storage.add_relationship(
                        source_id=memory_id,
                        target_id=s["id"],
                        relationship_type="derives",
                        confidence=float(s["similarity"]),
                    )
                    if rel_id:
                        relationships.append({
                            "id": str(rel_id),
                            "type": "derives",
                            "target_id": str(s["id"]),
                        })
            return relationships

        # With LLM, analyze relationships
        return await self._analyze_with_llm(memory, similar)

    async def _find_similar(
        self,
        content: str,
        container: str,
        exclude_id: Optional[UUID] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Find similar memories in the same container."""
        try:
            embedding = await self.embedding_provider.embed(content)
        except Exception:
            return []

        async with self.storage.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, memory_type,
                    1 - (embedding <=> $1::vector) as similarity
                FROM memories
                WHERE container = $2
                    AND is_latest = TRUE
                    AND embedding IS NOT NULL
                    AND id != $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                str(embedding),
                container,
                exclude_id,
                limit,
            )
            return [
                {**dict(r), "similarity": float(r["similarity"])}
                for r in rows
                if float(r["similarity"]) > 0.5
            ]

    async def _analyze_with_llm(
        self,
        memory: dict,
        similar: list[dict],
    ) -> list[dict]:
        """Use LLM to determine relationships."""
        similar_text = "\n".join(
            f"- ID: {s['id']} | Type: {s['memory_type']} | "
            f"Similarity: {s['similarity']:.2f} | Content: {s['content']}"
            for s in similar
        )

        prompt = (
            f"New memory: {memory['content']}\n"
            f"Type: {memory['memory_type']}\n\n"
            f"Existing similar memories:\n{similar_text}\n\n"
            f"What relationships exist between the new memory and existing ones?"
        )

        try:
            response = await self.llm.complete(prompt, RELATIONSHIP_SYSTEM_PROMPT)
            relationships_data = self._parse_response(response)

            created = []
            for rel in relationships_data:
                existing_id = rel.get("existing_memory_id")
                rel_type = rel.get("type")
                confidence = rel.get("confidence", 0.8)

                if not existing_id or not rel_type:
                    continue

                # Handle "updates" — mark old memory as not latest
                if rel_type == "updates":
                    await self.storage.update_memory(
                        UUID(existing_id), is_latest=False
                    )

                rel_id = await self.storage.add_relationship(
                    source_id=memory["id"],
                    target_id=UUID(existing_id),
                    relationship_type=rel_type,
                    confidence=confidence,
                    metadata={"reason": rel.get("reason", "")},
                )

                if rel_id:
                    created.append({
                        "id": str(rel_id),
                        "type": rel_type,
                        "target_id": existing_id,
                    })

            return created

        except Exception as e:
            logger.error("graph.llm_analysis_failed", error=str(e))
            return []

    def _parse_response(self, response: str) -> list[dict]:
        """Parse LLM response into relationship data."""
        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            return data.get("relationships", [])
        except (json.JSONDecodeError, ValueError):
            return []
