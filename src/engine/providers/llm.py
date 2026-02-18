from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, prompt: str, system: str = "") -> str:
        """Generate a completion from the LLM."""
        ...


def get_llm_provider(
    provider: str,
    model: str = "",
    **kwargs,
) -> LLMProvider:
    """Factory to create LLM provider by name."""
    if provider == "openai":
        from src.engine.providers.openai_provider import OpenAILLM

        return OpenAILLM(
            api_key=kwargs.get("api_key", ""),
            model=model or "gpt-4o-mini",
        )
    elif provider == "anthropic":
        from src.engine.providers.anthropic_provider import AnthropicLLM

        return AnthropicLLM(
            api_key=kwargs.get("api_key", ""),
            model=model or "claude-3-haiku-20240307",
        )
    elif provider == "ollama":
        from src.engine.providers.ollama_provider import OllamaLLM

        return OllamaLLM(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=model or "llama3",
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
