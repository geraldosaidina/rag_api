"""Application orchestration for grounded PFC question answering."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from rag.citations.citation_validator import CitationValidator
from rag.documents.models import PfcRecord
from rag.documents.store import PfcStore
from rag.context.evidence_assembler import (
    EvidenceAssembler,
    EvidenceAssemblerConfig,
    EvidenceDocument,
)
from rag.generation.answer_generator import AnswerGenerator, GenerationException
from rag.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverException
from rag.rerank.reranker import CrossEncoderReranker, RerankResult, RerankerException
from rag.service.models import AnswerDiagnostics, AnswerResult, SourceReference

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
        catalogue: PfcStore | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.evidence_assembler = evidence_assembler
        self.answer_generator = answer_generator
        self.citation_validator = citation_validator or CitationValidator()
        self.config = config or RAGServiceConfig()
        self.catalogue = catalogue

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

        records = self._catalogue_records(ranked)
        evidence = self.evidence_assembler.assemble(
            ranked,
            documents=_evidence_documents(records),
        )
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
        sources = _enrich_sources(citation_result.sources, records)
        total_ms = (time.perf_counter() - started) * 1000.0
        return AnswerResult(
            question=question,
            answer=citation_result.cleaned_answer,
            sources=sources,
            diagnostics=AnswerDiagnostics(
                total_duration_ms=total_ms,
                retrieval_candidate_count=len(candidates),
                evidence_chunk_count=len(evidence.items),
                used_llm_query_rewrite=used_rewrite,
                retrieval_duration_ms=retrieval_ms,
                rerank_duration_ms=rerank_ms,
                generation_duration_ms=generation_ms,
                # False means evidence was supplied to generation. It does not
                # mean every premise in the question was supported.
                insufficient_evidence=False,
                invalid_citations_removed=citation_result.invalid_citations,
            ),
        )

    def _catalogue_records(self, ranked: list[RerankResult]) -> dict[str, PfcRecord]:
        """One lookup for the distinct PFCs represented in the reranked candidates."""
        if self.catalogue is None:
            return {}
        pfc_ids: list[str] = []
        seen: set[str] = set()
        for result in ranked:
            raw_id = (result.metadata or {}).get("pfc_id")
            if not raw_id:
                continue
            pfc_id = str(raw_id)
            if pfc_id in seen:
                continue
            seen.add(pfc_id)
            pfc_ids.append(pfc_id)
        if not pfc_ids:
            return {}
        return {record.id: record for record in self.catalogue.find_by_ids(pfc_ids)}

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


def _evidence_documents(records: dict[str, PfcRecord]) -> dict[str, EvidenceDocument] | None:
    if not records:
        return None
    return {
        pfc_id: EvidenceDocument(
            pfc_id=record.id,
            title=record.title,
            authors=record.authors,
            year=record.year,
            institution=record.institution,
            course=record.course,
            department=record.department,
        )
        for pfc_id, record in records.items()
    }


def _enrich_sources(
    sources: tuple[SourceReference, ...],
    records: dict[str, PfcRecord],
) -> tuple[SourceReference, ...]:
    """Reuse the pre-generation catalogue records for cited sources."""
    if not sources or not records:
        return sources
    enriched = []
    for source in sources:
        record = records.get(source.pfc_id or "")
        if record is None:
            enriched.append(source)
            continue
        enriched.append(
            replace(
                source,
                pfc_id=record.id,
                title=record.title,
                authors=record.authors,
                year=record.year,
            )
        )
    return tuple(enriched)
