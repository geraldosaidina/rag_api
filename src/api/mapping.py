"""Map application AnswerResult objects to HTTP response schemas."""

from __future__ import annotations

from api.schemas import AskResponse, DiagnosticsResponse, SourceResponse
from rag.service.models import AnswerResult


def to_ask_response(result: AnswerResult) -> AskResponse:
    diagnostics = result.diagnostics
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceResponse(
                citation_id=source.citation_id,
                source=source.source,
                page=source.page,
                chunk_index=source.chunk_index,
                excerpt=source.excerpt,
            )
            for source in result.sources
        ],
        diagnostics=DiagnosticsResponse(
            total_duration_ms=diagnostics.total_duration_ms,
            retrieval_duration_ms=diagnostics.retrieval_duration_ms,
            rerank_duration_ms=diagnostics.rerank_duration_ms,
            generation_duration_ms=diagnostics.generation_duration_ms,
            retrieval_candidate_count=diagnostics.retrieval_candidate_count,
            evidence_chunk_count=diagnostics.evidence_chunk_count,
            used_llm_query_rewrite=diagnostics.used_llm_query_rewrite,
            insufficient_evidence=diagnostics.insufficient_evidence,
        ),
    )
