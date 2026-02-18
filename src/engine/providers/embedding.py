from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...

    def dimensions(self) -> int:
        """Return the number of dimensions for this model."""
        ...


def get_embedding_provider(
    provider: str,
    model: str = "",
    **kwargs,
) -> EmbeddingProvider:
    """Factory to create embedding provider by name."""
    if provider == "openai":
        from src.engine.providers.openai_provider import OpenAIEmbedding

        return OpenAIEmbedding(
            api_key=kwargs.get("api_key", ""),
            model=model or "text-embedding-3-small",
        )
    elif provider == "ollama":
        from src.engine.providers.ollama_provider import OllamaEmbedding

        return OllamaEmbedding(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=model or "nomic-embed-text",
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
