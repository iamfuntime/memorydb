from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import health, documents, memories, containers
from src.engine.storage import MemoryStorage
from src.utils.db import get_pool, close_pool
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging(app.state.log_level if hasattr(app.state, "log_level") else "info")
    logger.info("memorydb.starting")

    if hasattr(app.state, "database_url") and app.state.database_url:
        pool = await get_pool(app.state.database_url)
        app.state.storage = MemoryStorage(pool)
        logger.info("memorydb.db_connected")
    else:
        app.state.storage = None
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
