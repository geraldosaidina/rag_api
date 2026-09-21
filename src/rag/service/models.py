"""Application-level answer contracts for the RAG QA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceReference:
    """Traceable documentary source supporting part of an answer."""

    citation_id: str
    source: str
    page: int | str | None
    chunk_index: int | str | None
    chunk_id: str | None
    excerpt: str | None = None


@dataclass(frozen=True)
class AnswerDiagnostics:
    """Lightweight instrumentation for evaluation; never user-facing prose."""

    total_duration_ms: float
    retrieval_candidate_count: int
    evidence_chunk_count: int
    used_llm_query_rewrite: bool | None = None
    retrieval_duration_ms: float | None = None
    rerank_duration_ms: float | None = None
    generation_duration_ms: float | None = None
    insufficient_evidence: bool = False
    invalid_citations_removed: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerResult:
    """Structured application response returned by RAGService.answer()."""

    question: str
    answer: str
    sources: tuple[SourceReference, ...]
    diagnostics: AnswerDiagnostics
