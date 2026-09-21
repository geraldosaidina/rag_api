"""Application orchestration for grounded PFC question answering."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from rag.citations.citation_validator import CitationValidator
from rag.context.evidence_assembler import EvidenceAssembler, EvidenceAssemblerConfig
from rag.generation.answer_generator import AnswerGenerator, GenerationException
from rag.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverException
from rag.rerank.reranker import CrossEncoderReranker, RerankerException
from rag.service.models import AnswerDiagnostics, AnswerResult

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "O corpus documental disponível não contém informação suficiente para "
    "responder a esta pergunta de forma fiável."
)


@dataclass(frozen=True)
class RAGServiceConfig:
    candidate_k: int = 40
    rerank_top_k: int = 8
    max_evidence_chunks: int = 5
    max_evidence_chars: int = 6000


class RAGServiceException(Exception):
    """Raised for invalid input or infrastructure failures in the QA pipeline."""


class RAGService:
    """Thin orchestrator: retrieve → rerank → evidence → generate → cite."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        evidence_assembler: EvidenceAssembler,
        answer_generator: AnswerGenerator,
        citation_validator: CitationValidator | None = None,
        config: RAGServiceConfig | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.evidence_assembler = evidence_assembler
        self.answer_generator = answer_generator
        self.citation_validator = citation_validator or CitationValidator()
        self.config = config or RAGServiceConfig()

    def answer(self, question: str) -> AnswerResult:
        if not question or not question.strip():
            raise RAGServiceException("Cannot answer an empty question.")

        question = question.strip()
        started = time.perf_counter()
        retrieval_ms: float | None = None
        rerank_ms: float | None = None
        generation_ms: float | None = None
        used_rewrite: bool | None = None

        try:
            retrieve_started = time.perf_counter()
            candidates = self.retriever.retrieve(
                query=question,
                candidate_k=self.config.candidate_k,
            )
            retrieval_ms = (time.perf_counter() - retrieve_started) * 1000.0
        except HybridRetrieverException as exc:
            raise RAGServiceException(f"Retrieval failed: {exc}") from exc
        except Exception as exc:
            raise RAGServiceException(
                f"Unexpected retrieval failure: {type(exc).__name__}: {exc}"
            ) from exc

        if candidates:
            used_rewrite = bool(candidates[0].metadata.get("used_llm_query_rewrite"))

        if not candidates:
            return self._insufficient_result(
                question=question,
                started=started,
                candidate_count=0,
                evidence_count=0,
                used_rewrite=used_rewrite,
                retrieval_ms=retrieval_ms,
                rerank_ms=0.0,
                generation_ms=None,
            )

        try:
            rerank_started = time.perf_counter()
            ranked = self.reranker.rerank(
                query=question,
                candidates=candidates,
                top_k=self.config.rerank_top_k,
            )
            rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        except RerankerException as exc:
            raise RAGServiceException(f"Reranking failed: {exc}") from exc
        except Exception as exc:
            raise RAGServiceException(
                f"Unexpected reranking failure: {type(exc).__name__}: {exc}"
            ) from exc

        evidence = self.evidence_assembler.assemble(ranked)
        if evidence.is_empty:
            return self._insufficient_result(
                question=question,
                started=started,
                candidate_count=len(candidates),
                evidence_count=0,
                used_rewrite=used_rewrite,
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                generation_ms=None,
            )

        try:
            generation_started = time.perf_counter()
            raw_answer = self.answer_generator.generate(question, evidence)
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
        except GenerationException as exc:
            raise RAGServiceException(f"Generation failed: {exc}") from exc

        citation_result = self.citation_validator.validate(raw_answer, evidence)
        total_ms = (time.perf_counter() - started) * 1000.0
        return AnswerResult(
            question=question,
            answer=citation_result.cleaned_answer,
            sources=citation_result.sources,
            diagnostics=AnswerDiagnostics(
                total_duration_ms=total_ms,
                retrieval_candidate_count=len(candidates),
                evidence_chunk_count=len(evidence.items),
                used_llm_query_rewrite=used_rewrite,
                retrieval_duration_ms=retrieval_ms,
                rerank_duration_ms=rerank_ms,
                generation_duration_ms=generation_ms,
                insufficient_evidence=False,
                invalid_citations_removed=citation_result.invalid_citations,
            ),
        )

    def _insufficient_result(
        self,
        *,
        question: str,
        started: float,
        candidate_count: int,
        evidence_count: int,
        used_rewrite: bool | None,
        retrieval_ms: float | None,
        rerank_ms: float | None,
        generation_ms: float | None,
    ) -> AnswerResult:
        logger.info("Insufficient evidence for question: %s", question)
        return AnswerResult(
            question=question,
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=(),
            diagnostics=AnswerDiagnostics(
                total_duration_ms=(time.perf_counter() - started) * 1000.0,
                retrieval_candidate_count=candidate_count,
                evidence_chunk_count=evidence_count,
                used_llm_query_rewrite=used_rewrite,
                retrieval_duration_ms=retrieval_ms,
                rerank_duration_ms=rerank_ms,
                generation_duration_ms=generation_ms,
                insufficient_evidence=True,
            ),
        )
