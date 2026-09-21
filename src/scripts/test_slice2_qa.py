"""Deterministic Slice 2 regression tests for context, citations, and orchestration.

Run:
  uv run python src/scripts/test_slice2_qa.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.citations.citation_validator import CitationValidator
from rag.context.evidence_assembler import EvidenceAssembler, EvidenceAssemblerConfig
from rag.generation.answer_generator import AnswerGenerator, GenerationConfig, GenerationException
from rag.rerank.reranker import RerankResult
from rag.service.rag_service import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    RAGService,
    RAGServiceException,
)
from rag.vector.chroma_store import SearchResult


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rerank_result(content: str, *, source: str, page: int, chunk_index: int, chunk_id: str) -> RerankResult:
    return RerankResult(
        content=content,
        metadata={
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
        },
        original_score=0.5,
        rerank_score=0.9,
        answerability_score=0.0,
        final_score=0.9,
    )


def test_evidence_labels_are_deterministic() -> None:
    assembler = EvidenceAssembler(EvidenceAssemblerConfig(max_chunks=3, max_chars=5000))
    ranked = [
        _rerank_result("Alpha definition text.", source="a.pdf", page=1, chunk_index=0, chunk_id="a:0"),
        _rerank_result("Beta mechanism text.", source="a.pdf", page=2, chunk_index=1, chunk_id="a:1"),
    ]
    pack = assembler.assemble(ranked)
    _assert([i.citation_id for i in pack.items] == ["S1", "S2"], "Labels must be S1, S2 in rank order.")
    _assert("[S1]" in pack.context_text and "[S2]" in pack.context_text, "Context must include labels.")
    _assert("Source: a.pdf" in pack.context_text, "Context must preserve source.")
    _assert("Page: 1" in pack.context_text and "Page: 2" in pack.context_text, "Pages preserved.")
    _assert(pack.items[0].chunk_id == "a:0", "chunk_id preserved.")


def test_evidence_budget_and_dedupe() -> None:
    assembler = EvidenceAssembler(EvidenceAssemblerConfig(max_chunks=2, max_chars=2000))
    ranked = [
        _rerank_result("A" * 80, source="a.pdf", page=1, chunk_index=0, chunk_id="dup"),
        _rerank_result("B" * 80, source="a.pdf", page=1, chunk_index=0, chunk_id="dup"),
        _rerank_result("C" * 80, source="a.pdf", page=3, chunk_index=2, chunk_id="c"),
        _rerank_result("D" * 80, source="a.pdf", page=4, chunk_index=3, chunk_id="d"),
    ]
    pack = assembler.assemble(ranked)
    _assert(len(pack.items) == 2, f"Expected 2 chunks after budget/dedupe, got {len(pack.items)}")
    _assert(pack.items[0].chunk_id == "dup", "First unique chunk should be kept.")
    _assert(pack.items[1].chunk_id == "c", "Duplicate chunk_id must be skipped.")

    tight = EvidenceAssembler(EvidenceAssemblerConfig(max_chunks=5, max_chars=120))
    tight_pack = tight.assemble(ranked)
    _assert(len(tight_pack.items) <= 1, "Character budget must limit evidence count.")


def test_citation_validation_mapping_and_invalids() -> None:
    assembler = EvidenceAssembler()
    pack = assembler.assemble(
        [
            _rerank_result("Evidence one.", source="doc.pdf", page=10, chunk_index=0, chunk_id="c1"),
            _rerank_result("Evidence two.", source="doc.pdf", page=11, chunk_index=1, chunk_id="c2"),
        ]
    )
    validator = CitationValidator()
    result = validator.validate(
        "RAG combina recuperação e geração [S1][S2]. Facto inventado [S9] e outra [S1].",
        pack,
    )
    _assert(result.used_citation_ids == ("S1", "S2"), "Used IDs must be unique and ordered by first use.")
    _assert(result.invalid_citations == ("S9",), "Invalid ID must be reported.")
    _assert("[S9]" not in result.cleaned_answer, "Invalid citation must be stripped.")
    _assert(len(result.sources) == 2, "Duplicate [S1] must not duplicate SourceReference.")
    _assert(result.sources[0].page == 10 and result.sources[1].page == 11, "Pages must match evidence.")
    _assert(result.sources[0].source == "doc.pdf", "Source metadata is application-owned.")

    paren = validator.validate("Afirmação com formato alternativo (S1).", pack)
    _assert(paren.used_citation_ids == ("S1",), "Parenthetical (S1) must map to evidence.")
    _assert("[S1]" in paren.cleaned_answer, "Parenthetical citations should normalize to [S1].")


def test_generator_uses_assembled_evidence_and_portuguese_default() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = MagicMock(content="Resposta [S1].")
    generator = AnswerGenerator(llm_client=llm, config=GenerationConfig())
    pack = EvidenceAssembler().assemble(
        [_rerank_result("conteudo", source="x.pdf", page=1, chunk_index=0, chunk_id="x")]
    )
    answer = generator.generate("Como funciona X?", pack)
    _assert(answer == "Resposta [S1].", "Generator must return model content.")
    messages = llm.invoke_messages.call_args.args[0]
    system_prompt = messages[0][1]
    user_prompt = messages[1][1]
    _assert("Portuguese" in system_prompt, "Default response language must be Portuguese.")
    _assert("[S1]" in user_prompt and "conteudo" in user_prompt, "Only assembled evidence is supplied.")
    _assert("Como funciona X?" in user_prompt, "Question must be passed through.")


def test_orchestration_flow_and_insufficient_evidence() -> None:
    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()
    assembler = EvidenceAssembler(EvidenceAssemblerConfig(max_chunks=2))
    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        evidence_assembler=assembler,
        answer_generator=generator,
        citation_validator=CitationValidator(),
    )

    # Empty retrieval → insufficient evidence, no generation.
    retriever.retrieve.return_value = []
    empty = service.answer("Pergunta sem evidência?")
    _assert(empty.answer == INSUFFICIENT_EVIDENCE_ANSWER, "No-evidence answer required.")
    _assert(empty.diagnostics.insufficient_evidence is True, "Flag must be set.")
    _assert(empty.sources == (), "No sources when no evidence.")
    generator.generate.assert_not_called()
    reranker.rerank.assert_not_called()

    # Happy path flow order.
    retriever.retrieve.return_value = [
        SearchResult(
            content="Texto útil sobre RAG.",
            metadata={
                "source": "paper.pdf",
                "page": 5,
                "chunk_index": 0,
                "chunk_id": "paper:5:0",
                "used_llm_query_rewrite": True,
            },
            score=0.7,
        )
    ]
    reranker.rerank.return_value = [
        _rerank_result(
            "Texto útil sobre RAG.",
            source="paper.pdf",
            page=5,
            chunk_index=0,
            chunk_id="paper:5:0",
        )
    ]
    generator.generate.return_value = "RAG funciona com recuperação e geração [S1]."
    result = service.answer("Como funciona RAG?")
    _assert("RAG funciona" in result.answer, "Answer must come from generator.")
    _assert(len(result.sources) == 1, "One valid citation expected.")
    _assert(result.sources[0].citation_id == "S1", "Citation mapping required.")
    _assert(result.diagnostics.used_llm_query_rewrite is True, "Rewrite diagnostic expected.")
    _assert(result.diagnostics.retrieval_candidate_count == 1, "Candidate count diagnostic.")
    _assert(result.diagnostics.evidence_chunk_count == 1, "Evidence count diagnostic.")
    _assert(result.diagnostics.total_duration_ms >= 0, "Timing diagnostic required.")
    retriever.retrieve.assert_called()
    reranker.rerank.assert_called()
    generator.generate.assert_called()


def test_empty_question_and_generation_failure_are_distinct() -> None:
    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()
    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        evidence_assembler=EvidenceAssembler(),
        answer_generator=generator,
    )
    try:
        service.answer("   ")
        raise AssertionError("Empty question must raise RAGServiceException.")
    except RAGServiceException as exc:
        _assert("empty" in str(exc).lower(), "Empty-question message required.")

    retriever.retrieve.return_value = [
        SearchResult(
            content="x",
            metadata={"source": "a.pdf", "page": 1, "chunk_index": 0, "chunk_id": "a"},
            score=0.1,
        )
    ]
    reranker.rerank.return_value = [
        _rerank_result("x", source="a.pdf", page=1, chunk_index=0, chunk_id="a")
    ]
    generator.generate.side_effect = GenerationException("Ollama down")
    try:
        service.answer("Pergunta válida?")
        raise AssertionError("Generation failure must raise RAGServiceException.")
    except RAGServiceException as exc:
        _assert("Generation failed" in str(exc), "Infrastructure failure must remain distinct.")


def main() -> int:
    test_evidence_labels_are_deterministic()
    test_evidence_budget_and_dedupe()
    test_citation_validation_mapping_and_invalids()
    test_generator_uses_assembled_evidence_and_portuguese_default()
    test_orchestration_flow_and_insufficient_evidence()
    test_empty_question_and_generation_failure_are_distinct()
    print("All Slice 2 QA regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
