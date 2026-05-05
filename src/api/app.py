from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import init_system
from src.api.middleware import RequestIdMiddleware
from src.api.routes import router
from src.core.config_loader import settings
from src.core.logger import logger


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle hook."""
    logger.info("API starting up …")

    # LangSmith tracing is auto-enabled by LangChain when these env vars are set;
    # logging here makes it explicit whether the API is sending traces so a
    # silent misconfiguration doesn't go unnoticed in production.
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    if tracing_enabled and os.getenv("LANGSMITH_API_KEY"):
        logger.info(
            f"LangSmith tracing ENABLED (project={os.getenv('LANGSMITH_PROJECT', 'default')})."
        )
    else:
        logger.info("LangSmith tracing DISABLED.")

    system = init_system()
    await system.try_connect_existing()
    logger.info(f"Knowledge base ready: {system.is_ready} (store: {system.vector_store_type})")
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

    # Correlation IDs — must run before CORS so the header is present on
    # preflight responses too.
    application.add_middleware(RequestIdMiddleware)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        expose_headers=["X-Request-ID"],
    )

    application.include_router(router, prefix="/api")

    return application


# Module-level instance used by ``uvicorn src.api.app:app``
app = create_app()
