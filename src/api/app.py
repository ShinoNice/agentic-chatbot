from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import init_system
from src.api.routes import router
from src.core.logger import logger


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle hook."""
    logger.info("API starting up …")
    system = init_system()
    await system.try_connect_existing()
    logger.info(
        f"Knowledge base ready: {system.is_ready} "
        f"(store: {system.vector_store_type})"
    )
    yield
    logger.info("API shutting down.")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI instance."""
    application = FastAPI(
        title="Agentic RAG Chatbot API",
        description=(
            "Multi-agent Retrieval-Augmented Generation system with "
            "a self-correcting verification loop."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS (permissive for local dev) ───────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────
    application.include_router(router, prefix="/api")

    return application


# Module-level instance used by ``uvicorn src.api.app:app``
app = create_app()
