from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str

    # Embedding provider
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # LLM provider
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # API
    api_port: int = 8080
    api_token: Optional[str] = None
    log_level: str = "info"

    model_config = {"env_file": ".env", "case_sensitive": False}
