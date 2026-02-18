from src.config import Settings


def test_settings_defaults():
    s = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        openai_api_key="sk-test",
    )
    assert s.embedding_provider == "openai"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.log_level == "info"
    assert s.api_port == 8080


def test_settings_overrides():
    s = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        llm_provider="anthropic",
        llm_model="claude-3-haiku-20240307",
        log_level="debug",
        api_port=9090,
    )
    assert s.embedding_provider == "ollama"
    assert s.llm_provider == "anthropic"
    assert s.api_port == 9090
