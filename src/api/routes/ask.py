"""Ask endpoint: thin HTTP adapter over RAGService."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.mapping import to_ask_response
from api.schemas import AskRequest, AskResponse, ErrorResponse
from rag.service.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ask"])


def _get_rag_service(request: Request) -> RAGService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "detail": "The question-answering service is not ready.",
            },
        )
    return service


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """
    Answer a natural-language question using the reusable RAGService.

    Declared as a synchronous route so FastAPI executes the blocking RAG pipeline
    in its threadpool instead of blocking the event loop.
    """
    service = _get_rag_service(request)
    logger.info("Handling /api/v1/ask request.")
    result = service.answer(payload.question)
    return to_ask_response(result)
