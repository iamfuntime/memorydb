from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import health, documents, memories, containers, search, profile, stats
from src.config import Settings
from src.engine.storage import MemoryStorage
from src.engine.search import HybridSearch
from src.engine.pipeline import ProcessingPipeline
from src.engine.providers.embedding import get_embedding_provider
from src.engine.providers.llm import get_llm_provider
from src.utils.db import get_pool, close_pool
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load settings from environment
    try:
        settings = Settings()
        for key, value in settings.model_dump().items():
            setattr(app.state, key, value)
    except Exception:
        pass

    setup_logging(app.state.log_level if hasattr(app.state, "log_level") else "info")
    logger.info("memorydb.starting")

    if hasattr(app.state, "database_url") and app.state.database_url:
        pool = await get_pool(app.state.database_url)
        app.state.storage = MemoryStorage(pool)

        # Initialize embedding provider and search engine
        emb_provider = getattr(app.state, "embedding_provider", getattr(app.state, "embedding_provider_name", "openai"))
        emb_model = getattr(app.state, "embedding_model", "text-embedding-3-small")
        emb_kwargs = {}
        if hasattr(app.state, "openai_api_key") and app.state.openai_api_key:
            emb_kwargs["api_key"] = app.state.openai_api_key
        if hasattr(app.state, "ollama_base_url"):
            emb_kwargs["base_url"] = app.state.ollama_base_url

        try:
            embedding_provider = get_embedding_provider(
                emb_provider, model=emb_model, **emb_kwargs
            )
            app.state.search_engine = HybridSearch(pool, embedding_provider)
            app.state.embedding_provider = embedding_provider
            logger.info("memorydb.search_engine_initialized", provider=emb_provider)
        except Exception as e:
            logger.warning("memorydb.search_engine_failed", error=str(e))
            app.state.search_engine = None
            app.state.embedding_provider = None

        # Initialize LLM provider (optional)
        llm_provider_name = getattr(app.state, "llm_provider", getattr(app.state, "llm_provider_name", "openai"))
        llm_model = getattr(app.state, "llm_model", "gpt-4o-mini")
        llm_kwargs = {}
        if hasattr(app.state, "openai_api_key") and app.state.openai_api_key:
            llm_kwargs["api_key"] = app.state.openai_api_key
        if hasattr(app.state, "anthropic_api_key") and app.state.anthropic_api_key:
            llm_kwargs["api_key"] = app.state.anthropic_api_key
        if hasattr(app.state, "ollama_base_url"):
            llm_kwargs["base_url"] = app.state.ollama_base_url

        try:
            llm_provider = get_llm_provider(
                llm_provider_name, model=llm_model, **llm_kwargs
            )
            logger.info("memorydb.llm_provider_initialized", provider=llm_provider_name)
        except Exception as e:
            logger.warning("memorydb.llm_provider_failed", error=str(e))
            llm_provider = None

        # Initialize processing pipeline
        if app.state.embedding_provider:
            app.state.pipeline = ProcessingPipeline(
                app.state.storage, app.state.embedding_provider, llm_provider
            )
            logger.info("memorydb.pipeline_initialized")
        else:
            app.state.pipeline = None

        logger.info("memorydb.db_connected")
    else:
        app.state.storage = None
        app.state.search_engine = None
        app.state.embedding_provider = None
        app.state.pipeline = None
        logger.warning("memorydb.no_database_url")

    yield

    # Shutdown
    await close_pool()
    logger.info("memorydb.shutdown")


app = FastAPI(
    title="MemoryDB",
    description="Self-hosted AI agent memory system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router, prefix="/v1/documents", tags=["documents"])
app.include_router(memories.router, prefix="/v1/memories", tags=["memories"])
app.include_router(containers.router, prefix="/v1/containers", tags=["containers"])
app.include_router(search.router, prefix="/v1/search", tags=["search"])
app.include_router(profile.router, prefix="/v1/profile", tags=["profile"])
app.include_router(stats.router, prefix="/v1/stats", tags=["stats"])
