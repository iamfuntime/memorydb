import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Dimension map for common Ollama embedding models
_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


class OllamaEmbedding:
    """Ollama embedding provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimensions = _DIMENSIONS.get(model, 768)

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]

    def dimensions(self) -> int:
        return self._dimensions


class OllamaLLM:
    """Ollama LLM provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(self, prompt: str, system: str = "") -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["response"]
