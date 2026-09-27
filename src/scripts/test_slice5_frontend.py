"""Slice 5 checks for the student frontend shell and evaluation log.

Run:
  uv run python src/scripts/test_slice5_frontend.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from api.app import create_app
from rag.service.models import AnswerDiagnostics, AnswerResult, SourceReference


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _client(service: MagicMock, **kwargs: object) -> TestClient:
    app = create_app(rag_service=service, initialize_rag=False, **kwargs)
    return TestClient(app, raise_server_exceptions=False)


def test_pages_render_student_shell() -> None:
    client = _client(MagicMock())
    assistant = client.get("/")
    library = client.get("/biblioteca")
    styles = client.get("/assets/styles.css")
    _assert(assistant.status_code == 200, "Assistant page must load.")
    _assert("Explore os Projectos Finais de Curso" in assistant.text, "Assistant purpose.")
    _assert('id="question"' in assistant.text, "Question field.")
    _assert('id="status"' in assistant.text and "aria-live" in assistant.text, "Loading status.")
    _assert("retrieval_duration" not in assistant.text, "Metrics stay off the assistant page.")
    _assert(library.status_code == 200, "Library page must load.")
    _assert('id="catalogue-search"' in library.text, "Library search.")
    _assert("Abrir PFC" in library.text, "Original PDF action.")
    _assert(styles.status_code == 200, "Stylesheet must load.")
    _assert("@media (min-width: 800px)" in styles.text, "Desktop layout adjustment exists.")
    _assert("overflow-wrap: anywhere" in styles.text, "Long titles wrap.")


def test_cors_is_explicit_and_not_wildcard() -> None:
    allowed = _client(MagicMock(), cors_origins=["http://127.0.0.1:4173"])
    response = allowed.get("/", headers={"Origin": "http://127.0.0.1:4173"})
    _assert(
        response.headers.get("access-control-allow-origin") == "http://127.0.0.1:4173",
        "Configured origin is allowed.",
    )
    closed = _client(MagicMock(), cors_origins=[])
    denied = closed.get("/", headers={"Origin": "http://evil.example"})
    _assert(
        denied.headers.get("access-control-allow-origin") is None,
        "Unconfigured origins are not reflected.",
    )
    _assert("*" not in (response.headers.get("access-control-allow-origin") or ""), "No wildcard origin.")


def test_evaluation_log_keeps_metrics_only() -> None:
    service = MagicMock()
    service.answer.return_value = AnswerResult(
        question="PERGUNTA-SECRETA",
        answer="RESPOSTA-SECRETA [S1]",
        sources=(
            SourceReference(
                citation_id="S1",
                source="edson.pdf",
                page=74,
                chunk_index=1,
                chunk_id="chunk-secreto",
                excerpt="EXCERTO-SECRETO",
                pfc_id="pfc-edson",
                title="RFID",
                authors=("Edson Samuel Langa",),
                year=2018,
            ),
        ),
        diagnostics=AnswerDiagnostics(
            total_duration_ms=1000.0,
            retrieval_candidate_count=40,
            evidence_chunk_count=5,
            retrieval_duration_ms=20.0,
            rerank_duration_ms=10.0,
            generation_duration_ms=900.0,
            insufficient_evidence=False,
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "interactions.jsonl"
        client = _client(service, evaluation_log_path=str(path))
        response = client.post("/api/v1/ask", json={"question": "PERGUNTA-SECRETA"})
        _assert(response.status_code == 200, "Ask still succeeds when logging.")
        text = path.read_text(encoding="utf-8")
        for secret in ("PERGUNTA-SECRETA", "RESPOSTA-SECRETA", "EXCERTO-SECRETO", "chunk-secreto"):
            _assert(secret not in text, f"Log must not contain {secret}.")
        record = json.loads(text)
        _assert(record["source_count"] == 1, "Source count recorded.")
        _assert(record["retrieval_candidate_count"] == 40, "Candidate count recorded.")
        _assert(record["evidence_chunk_count"] == 5, "Evidence count recorded.")
        _assert(record["insufficient_evidence"] is False, "Insufficient flag recorded.")
        _assert(record["total_duration_ms"] == 1000.0, "Total duration recorded.")
        _assert("interaction_id" in record and "timestamp" in record, "Anonymous id and time.")


def main() -> int:
    test_pages_render_student_shell()
    test_cors_is_explicit_and_not_wildcard()
    test_evaluation_log_keeps_metrics_only()
    print("All Slice 5 frontend checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
