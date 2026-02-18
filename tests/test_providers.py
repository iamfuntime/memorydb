import pytest

from src.engine.providers.embedding import get_embedding_provider
from src.engine.providers.llm import get_llm_provider


def test_get_openai_embedding_provider():
    provider = get_embedding_provider("openai", model="text-embedding-3-small", api_key="sk-test")
    assert provider.dimensions() == 1536


def test_get_openai_large_embedding_provider():
    provider = get_embedding_provider("openai", model="text-embedding-3-large", api_key="sk-test")
    assert provider.dimensions() == 3072


def test_get_ollama_embedding_provider():
    provider = get_embedding_provider("ollama", model="nomic-embed-text")
    assert provider.dimensions() == 768


def test_get_unknown_embedding_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_provider("nonexistent")


def test_get_openai_llm_provider():
    provider = get_llm_provider("openai", model="gpt-4o-mini", api_key="sk-test")
    assert hasattr(provider, "complete")


def test_get_anthropic_llm_provider():
    provider = get_llm_provider("anthropic", model="claude-3-haiku-20240307", api_key="sk-test")
    assert hasattr(provider, "complete")


def test_get_ollama_llm_provider():
    provider = get_llm_provider("ollama", model="llama3")
    assert hasattr(provider, "complete")


def test_get_unknown_llm_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider("nonexistent")
