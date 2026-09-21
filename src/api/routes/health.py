"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def liveness() -> HealthResponse:
    """Process is alive. Does not touch RAG models or Chroma."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def readiness(request: Request) -> JSONResponse:
    """
    Application can serve RAG questions only when RAGService initialized successfully.

    Performs a lightweight Chroma reachability check when the service is available.
    Does not invoke LLM generation.
    """
    ready = bool(getattr(request.app.state, "ready", False))
    service = getattr(request.app.state, "rag_service", None)
    init_error = getattr(request.app.state, "init_error", None)

    if not ready or service is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": init_error or "RAG service is not initialized.",
            },
        )

    try:
        vector_store = service.retriever.vector_store
        if hasattr(vector_store, "health_check") and not vector_store.health_check():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "detail": "Vector store health check failed.",
                },
            )
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": f"Dependency check failed: {type(exc).__name__}",
            },
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready", "detail": None},
    )
