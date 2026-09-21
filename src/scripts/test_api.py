"""Deterministic FastAPI tests for Slice 3.

Run:
  uv run python src/scripts/test_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from api.app import create_app
from rag.service.models import AnswerDiagnostics, AnswerResult, SourceReference
from rag.service.rag_service import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    RAGServiceException,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sample_result(*, insufficient: bool = False) -> AnswerResult:
    if insufficient:
        return AnswerResult(
            question="Pergunta sem suporte?",
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=(),
            diagnostics=AnswerDiagnostics(
                total_duration_ms=12.5,
                retrieval_candidate_count=0,
                evidence_chunk_count=0,
                used_llm_query_rewrite=False,
                retrieval_duration_ms=5.0,
                rerank_duration_ms=0.0,
                generation_duration_ms=None,
                insufficient_evidence=True,
            ),
        )
    return AnswerResult(
        question="Como funciona RAG?",
        answer="RAG combina recuperação e geração [S1].",
        sources=(
            SourceReference(
                citation_id="S1",
                source="paper.pdf",
                page=3,
                chunk_index=1,
                chunk_id="paper:3:1",
                excerpt="RAG combines retrieval and generation.",
            ),
        ),
        diagnostics=AnswerDiagnostics(
            total_duration_ms=120.0,
            retrieval_candidate_count=40,
            evidence_chunk_count=5,
            used_llm_query_rewrite=True,
            retrieval_duration_ms=30.0,
            rerank_duration_ms=15.0,
            generation_duration_ms=70.0,
            insufficient_evidence=False,
            invalid_citations_removed=("S9",),
        ),
    )


def _client_with_service(service: MagicMock) -> TestClient:
    app = create_app(rag_service=service, initialize_rag=False)
    return TestClient(app, raise_server_exceptions=False)


def test_ask_success_schema() -> None:
    service = MagicMock()
    service.answer.return_value = _sample_result()
    client = _client_with_service(service)

    response = client.post("/api/v1/ask", json={"question": "Como funciona RAG?"})
    _assert(response.status_code == 200, f"Expected 200, got {response.status_code}")
    body = response.json()
    _assert(body["answer"].startswith("RAG combina"), "Answer missing.")
    _assert(body["sources"][0]["citation_id"] == "S1", "Source citation missing.")
    _assert(body["sources"][0]["source"] == "paper.pdf", "Source filename missing.")
    _assert(body["sources"][0]["page"] == 3, "Page missing.")
    _assert("chunk_id" not in body["sources"][0], "chunk_id must not leak to public API.")
    _assert(body["diagnostics"]["insufficient_evidence"] is False, "Flag wrong.")
    _assert(body["diagnostics"]["retrieval_candidate_count"] == 40, "Diagnostics missing.")
    _assert(
        "invalid_citations_removed" not in body["diagnostics"],
        "Internal diagnostics must stay internal.",
    )
    service.answer.assert_called_once_with("Como funciona RAG?")


def test_blank_and_missing_question_rejected() -> None:
    service = MagicMock()
    client = _client_with_service(service)

    blank = client.post("/api/v1/ask", json={"question": "   "})
    _assert(blank.status_code == 422, f"Blank question expected 422, got {blank.status_code}")
    _assert(blank.json()["error"] == "validation_error", "Validation error contract.")

    missing = client.post("/api/v1/ask", json={})
    _assert(missing.status_code == 422, f"Missing question expected 422, got {missing.status_code}")
    service.answer.assert_not_called()


def test_insufficient_evidence_is_http_200() -> None:
    service = MagicMock()
    service.answer.return_value = _sample_result(insufficient=True)
    client = _client_with_service(service)

    response = client.post(
        "/api/v1/ask",
        json={"question": "Qual é a capital da Austrália segundo os PFCs?"},
    )
    _assert(response.status_code == 200, "Insufficient evidence must be HTTP 200.")
    body = response.json()
    _assert(body["diagnostics"]["insufficient_evidence"] is True, "Flag required.")
    _assert(body["sources"] == [], "No fabricated sources.")
    _assert("não contém informação suficiente" in body["answer"], "Abstention text.")


def test_infrastructure_and_unexpected_failures() -> None:
    service = MagicMock()
    client = _client_with_service(service)

    service.answer.side_effect = RAGServiceException("Generation failed: Ollama down")
    unavailable = client.post("/api/v1/ask", json={"question": "Como funciona RAG?"})
    _assert(unavailable.status_code == 503, f"Expected 503, got {unavailable.status_code}")
    _assert(unavailable.json()["error"] == "service_unavailable", "503 contract.")
    _assert("Ollama" not in unavailable.json()["detail"], "No internal details.")

    service.answer.side_effect = RuntimeError("secret stack boom")
    boom = client.post("/api/v1/ask", json={"question": "Como funciona RAG?"})
    _assert(boom.status_code == 500, f"Expected 500, got {boom.status_code}")
    body = boom.json()
    _assert(body["error"] == "internal_error", "500 contract.")
    _assert("secret stack boom" not in body["detail"], "Stack/message must not leak.")


def test_health_and_readiness() -> None:
    service = MagicMock()
    vector_store = MagicMock()
    vector_store.health_check.return_value = True
    service.retriever.vector_store = vector_store
    client = _client_with_service(service)

    health = client.get("/health")
    _assert(health.status_code == 200, "Liveness must work.")
    _assert(health.json()["status"] == "ok", "Liveness payload.")

    ready = client.get("/ready")
    _assert(ready.status_code == 200, "Readiness should be ready with injected service.")
    _assert(ready.json()["status"] == "ready", "Ready status.")

    unready_app = create_app(rag_service=None, initialize_rag=False)
    unready_app.state.ready = False
    unready_app.state.init_error = "init failed"
    unready_client = TestClient(unready_app)
    not_ready = unready_client.get("/ready")
    _assert(not_ready.status_code == 503, "Unready must be 503.")
    ask_unready = unready_client.post("/api/v1/ask", json={"question": "test?"})
    _assert(ask_unready.status_code == 503, "Ask while unready must be 503.")


def test_rag_service_reused_across_requests() -> None:
    service = MagicMock()
    service.answer.return_value = _sample_result()
    client = _client_with_service(service)

    client.post("/api/v1/ask", json={"question": "Pergunta um?"})
    client.post("/api/v1/ask", json={"question": "Pergunta dois?"})
    _assert(service.answer.call_count == 2, "Same service instance must handle both requests.")


def main() -> int:
    test_ask_success_schema()
    test_blank_and_missing_question_rejected()
    test_insufficient_evidence_is_http_200()
    test_infrastructure_and_unexpected_failures()
    test_health_and_readiness()
    test_rag_service_reused_across_requests()
    print("All Slice 3 API regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
