"""Non-regression checks for Phase 2 generation benchmark hooks.

Run:
  uv run python src/scripts/test_benchmark_phase2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
ROOT = PROJECT_SRC.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.mapping import to_ask_response
from evaluation.metrics.generation import abstention_signals, score_citations
from rag.service.models import (
    AnswerDiagnostics,
    AnswerResult,
    EvidenceTraceItem,
    SourceReference,
)
from rag.service.rag_service import RAGService


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_api_mapping_omits_evidence_trace() -> None:
    result = AnswerResult(
        question="Q?",
        answer="Resposta [S1].",
        sources=(
            SourceReference(
                citation_id="S1",
                source="a.pdf",
                page=1,
                chunk_index=0,
                chunk_id="id",
                excerpt="trecho",
                pfc_id="pfc-1",
                title="Titulo",
                authors=("Autor",),
                year=2020,
            ),
        ),
        diagnostics=AnswerDiagnostics(
            total_duration_ms=1.0,
            retrieval_candidate_count=1,
            evidence_chunk_count=1,
            insufficient_evidence=False,
        ),
        evidence_context_text="SECRET_EVIDENCE_CONTEXT",
        evidence_items=(
            EvidenceTraceItem(
                citation_id="S1",
                content="SECRET_FULL_CONTENT",
                source="a.pdf",
                page=1,
                chunk_index=0,
                chunk_id="id",
                excerpt="trecho",
                pfc_id="pfc-1",
            ),
        ),
        raw_answer_before_citation_cleanup="raw",
    )
    payload = to_ask_response(result).model_dump()
    serialized = str(payload)
    _assert("SECRET_EVIDENCE_CONTEXT" not in serialized, "Evidence context must not reach API.")
    _assert("SECRET_FULL_CONTENT" not in serialized, "Evidence content must not reach API.")
    _assert("raw_answer_before_citation_cleanup" not in payload, "Raw answer not in API schema.")
    _assert("evidence_items" not in payload, "Evidence items not in API schema.")
    _assert(payload["sources"][0]["pfc_id"] == "pfc-1", "Public sources still mapped.")


def test_answer_result_defaults_keep_production_shape() -> None:
    result = AnswerResult(
        question="Q?",
        answer="A",
        sources=(),
        diagnostics=AnswerDiagnostics(
            total_duration_ms=1.0,
            retrieval_candidate_count=0,
            evidence_chunk_count=0,
            insufficient_evidence=True,
        ),
    )
    _assert(result.evidence_context_text is None, "Default evidence_context_text is None.")
    _assert(result.evidence_items == (), "Default evidence_items is empty.")
    _assert(result.raw_answer_before_citation_cleanup is None, "Default raw answer is None.")


def test_citation_metrics_supported_and_unsupported() -> None:
    sources = [
        {"citation_id": "S1", "pfc_id": "doc-a", "page": 10},
        {"citation_id": "S2", "pfc_id": "doc-b", "page": 3},
    ]
    supported = score_citations(
        sources=sources,
        primary_pfc_ids=["doc-a"],
        gold_pages_by_pfc={"doc-a": [10, 11], "doc-b": [1]},
        unsupported=False,
        invalid_citations_removed=("S9",),
    )
    _assert(supported["citation_doc_precision"] == 0.5, "One of two cited docs is primary.")
    _assert(supported["citation_doc_recall"] == 1.0, "Only primary doc was recalled.")
    _assert(supported["citation_page_precision"] == 0.5, "One page hit among evaluable.")
    _assert(supported["invalid_citation_count"] == 1, "Invalid citation counted.")

    unsupported = score_citations(
        sources=sources,
        primary_pfc_ids=[],
        gold_pages_by_pfc={},
        unsupported=True,
        invalid_citations_removed=(),
    )
    _assert(unsupported["citation_doc_precision"] is None, "Unsupported has no doc precision.")
    _assert(unsupported["unsupported_has_sources"] is True, "Sources flag for unsupported.")


def test_abstention_signals() -> None:
    signals = abstention_signals(
        answer="As evidências recuperadas não estabelecem esses documentos.",
        insufficient_evidence=False,
        unsupported=True,
    )
    _assert(signals["has_abstention_phrase"] is True, "Portuguese abstention phrase detected.")


def test_rag_service_still_exports_answer() -> None:
    _assert(callable(RAGService.answer), "RAGService.answer must remain the production entrypoint.")


def main() -> int:
    test_api_mapping_omits_evidence_trace()
    test_answer_result_defaults_keep_production_shape()
    test_citation_metrics_supported_and_unsupported()
    test_abstention_signals()
    test_rag_service_still_exports_answer()
    print("Phase 2 benchmark non-regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
