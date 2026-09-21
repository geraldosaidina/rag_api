"""HTTP error helpers and application exception mapping."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag.service.rag_service import RAGServiceException

logger = logging.getLogger(__name__)


def _error_body(error: str, detail: str) -> dict[str, str]:
    return {"error": error, "detail": detail}


def map_rag_service_exception(exc: RAGServiceException) -> JSONResponse:
    message = str(exc)
    lower = message.lower()

    if "empty question" in lower:
        return JSONResponse(
            status_code=400,
            content=_error_body("invalid_request", "Question must not be empty."),
        )

    if any(
        marker in lower
        for marker in (
            "retrieval failed",
            "reranking failed",
            "generation failed",
            "unexpected retrieval failure",
            "unexpected reranking failure",
        )
    ):
        logger.error("RAG infrastructure failure: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=503,
            content=_error_body(
                "service_unavailable",
                "The question-answering service is temporarily unavailable.",
            ),
        )

    logger.error("RAG application failure: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=_error_body(
            "internal_error",
            "An internal application error occurred while answering the question.",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        _ = request
        logger.info("Request validation failed: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content=_error_body(
                "validation_error",
                "Invalid request. Provide a non-empty question string.",
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        _ = request
        if isinstance(exc.detail, dict) and "error" in exc.detail and "detail" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body("http_error", str(exc.detail)),
        )

    @app.exception_handler(RAGServiceException)
    async def rag_service_exception_handler(
        request: Request, exc: RAGServiceException
    ) -> JSONResponse:
        _ = request
        return map_rag_service_exception(exc)

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        _ = request
        logger.exception("Unexpected API failure: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_error_body(
                "internal_error",
                "An unexpected internal error occurred.",
            ),
        )
