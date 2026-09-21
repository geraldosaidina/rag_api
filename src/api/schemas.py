"""HTTP request/response schemas for the PFC Assistant API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class SourceResponse(BaseModel):
    citation_id: str
    source: str
    page: int | str | None = None
    chunk_index: int | str | None = None
    excerpt: str | None = None


class DiagnosticsResponse(BaseModel):
    """Public diagnostics useful for MVP evaluation. Omits internal-only fields."""

    total_duration_ms: float
    retrieval_duration_ms: float | None = None
    rerank_duration_ms: float | None = None
    generation_duration_ms: float | None = None
    retrieval_candidate_count: int
    evidence_chunk_count: int
    used_llm_query_rewrite: bool | None = None
    insufficient_evidence: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    diagnostics: DiagnosticsResponse


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    detail: str | None = None
