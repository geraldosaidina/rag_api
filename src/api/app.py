"""FastAPI application factory and lifecycle for the PFC Assistant API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.errors import register_exception_handlers
from api.routes import ask, documents, health
from rag.documents.ingestion import recover_processing_documents
from rag.service.factory import RAGAppConfig, build_rag_service, load_rag_app_config
from rag.documents.store import PfcStore
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
            service = build_rag_service(config)
            app.state.rag_service = service
            app.state.pfc_store = service.catalogue
            app.state.pfc_corpus_directory = config.pfc_corpus_directory
            app.state.ready = True
            app.state.init_error = None
            recover_processing_documents(
                service.catalogue,
                service.retriever.vector_store,
            )
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
    pfc_store: PfcStore | None = None,
    pfc_corpus_directory: str | None = None,
    initialize_rag: bool = True,
    evaluation_log_path: str | None = None,
    cors_origins: list[str] | None = None,
    frontend_directory: str | None = None,
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
    app.state.pfc_store = pfc_store
    app.state.pfc_corpus_directory = pfc_corpus_directory
    app.state.initialize_rag = initialize_rag and rag_service is None
    app.state.ready = rag_service is not None
    app.state.init_error = None
    app.state.evaluation_log_path = evaluation_log_path

    origins = cors_origins if cors_origins is not None else _cors_origins_from_env()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ask.router)
    app.include_router(documents.router)
    _mount_frontend(app, frontend_directory)
    return app


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _mount_frontend(app: FastAPI, frontend_directory: str | None) -> None:
    directory = (
        Path(frontend_directory)
        if frontend_directory
        else Path(__file__).resolve().parents[2] / "frontend"
    )
    index = directory / "index.html"
    library = directory / "biblioteca.html"
    assets = directory / "assets"
    if not index.is_file() or not library.is_file() or not assets.is_dir():
        logger.info("Student frontend was not mounted because files are missing.")
        return

    @app.get("/", include_in_schema=False)
    def assistant_page() -> FileResponse:
        return FileResponse(index)

    @app.get("/biblioteca", include_in_schema=False)
    def library_page() -> FileResponse:
        return FileResponse(library)

    app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")
