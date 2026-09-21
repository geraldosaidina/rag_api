"""FastAPI application factory and lifecycle for the PFC Assistant API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.errors import register_exception_handlers
from api.routes import ask, health
from rag.service.factory import RAGAppConfig, build_rag_service, load_rag_app_config
from rag.service.rag_service import RAGService

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if getattr(app.state, "initialize_rag", True) and app.state.rag_service is None:
        logger.info("Initializing reusable RAGService for FastAPI lifecycle.")
        try:
            config: RAGAppConfig = app.state.rag_config or load_rag_app_config()
            app.state.rag_service = build_rag_service(config)
            app.state.ready = True
            app.state.init_error = None
            logger.info("RAGService initialized successfully.")
        except Exception as exc:
            app.state.rag_service = None
            app.state.ready = False
            app.state.init_error = f"{type(exc).__name__}: {exc}"
            logger.exception("RAGService initialization failed: %s", exc)
    elif app.state.rag_service is not None:
        app.state.ready = True
        app.state.init_error = None
        logger.info("Using preconfigured RAGService instance.")

    yield

    logger.info("Shutting down FastAPI application.")
    app.state.rag_service = None
    app.state.ready = False


def create_app(
    *,
    rag_service: RAGService | None = None,
    rag_config: RAGAppConfig | None = None,
    initialize_rag: bool = True,
) -> FastAPI:
    """
    Create the FastAPI application.

    For production/local serving, call with defaults so lifespan builds RAGService once.
    For deterministic tests, pass a mock rag_service and initialize_rag=False.
    """
    _configure_logging()

    app = FastAPI(
        title="PFC Assistant API",
        version="0.1.0",
        description=(
            "HTTP API for semantic exploration of Projectos Finais de Curso (PFCs) "
            "using Retrieval-Augmented Generation."
        ),
        lifespan=lifespan,
    )
    app.state.rag_service = rag_service
    app.state.rag_config = rag_config
    app.state.initialize_rag = initialize_rag and rag_service is None
    app.state.ready = rag_service is not None
    app.state.init_error = None

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ask.router)
    return app
