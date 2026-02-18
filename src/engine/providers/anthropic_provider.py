from anthropic import AsyncAnthropic

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicLLM:
    """Anthropic LLM provider."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def complete(self, prompt: str, system: str = "") -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system if system else "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
