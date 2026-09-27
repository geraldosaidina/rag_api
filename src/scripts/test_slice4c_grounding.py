"""Deterministic checks that catalogue metadata grounds generation.

Run:
  uv run python src/scripts/test_slice4c_grounding.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.citations.citation_validator import CitationValidator
from rag.context.evidence_assembler import EvidenceAssembler, EvidenceDocument
from rag.documents.models import PfcRecord, PfcStatus
from rag.generation.answer_generator import AnswerGenerator, GenerationConfig
from rag.rerank.reranker import RerankResult
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.service.rag_service import RAGService
from rag.vector.chroma_store import SearchResult


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _record(
    pfc_id: str,
    *,
    title: str | None,
    authors: tuple[str, ...] = (),
    year: int | None = None,
    institution: str | None = None,
    course: str | None = None,
    department: str | None = None,
) -> PfcRecord:
    return PfcRecord(
        id=pfc_id,
        title=title,
        authors=authors,
        year=year,
        course=course,
        institution=institution,
        department=department,
        supervisor=None,
        abstract=None,
        keywords=(),
        language="pt",
        original_filename=f"{pfc_id}.pdf",
        content_hash="a" * 64,
        storage_path=f"data/corpus/{pfc_id}.pdf",
        page_count=1,
        status=PfcStatus.INDEXED,
        error=None,
        created_at="",
        updated_at="",
    )


def _ranked(content: str, pfc_id: str, page: int) -> RerankResult:
    return RerankResult(
        content=content,
        metadata={
            "source": f"{pfc_id}.pdf",
            "page": page,
            "chunk_index": 0,
            "chunk_id": f"{pfc_id}:page-{page}:chunk-0:abc",
            "pfc_id": pfc_id,
        },
        original_score=0.4,
        rerank_score=0.8,
        answerability_score=0.0,
        final_score=0.8,
    )


def test_metadata_stays_with_its_citation_label() -> None:
    documents = {
        "pfc-y": EvidenceDocument(
            pfc_id="pfc-y",
            title="Projecto Y",
            authors=("Maria Costa",),
            year=2021,
            institution="Universidade Y",
            course="Engenharia Informática",
            department="Departamento de Informática",
        ),
        "pfc-z": EvidenceDocument(
            pfc_id="pfc-z",
            title="Projecto Z",
            authors=("Ana Lopes",),
            year=2018,
            institution="Universidade Z",
            course="Gestão",
        ),
    }
    pack = EvidenceAssembler().assemble(
        [
            _ranked("Texto do projecto Y.", "pfc-y", 3),
            _ranked("Texto do projecto Z.", "pfc-z", 9),
        ],
        documents=documents,
    )
    context = pack.context_text
    _assert(
        "Institutions established by the document metadata in this evidence: "
        "Universidade Y; Universidade Z" in context,
        "Scope note lists only the catalogue institutions.",
    )
    _assert(
        "Do not describe these sources as belonging to any other institution." in context,
        "Scope note forbids reassignment.",
    )
    blocks = [part for part in context.split("\n\n") if part.startswith("[S")]
    y_block, z_block = blocks
    _assert(y_block.startswith("[S1]"), "First evidence block is S1.")
    _assert("Institution: Universidade Y" in y_block, "S1 keeps its institution.")
    _assert("Authors: Maria Costa" in y_block, "S1 keeps its author.")
    _assert("Year: 2021" in y_block, "S1 keeps its year.")
    _assert("Course: Engenharia Informática" in y_block, "S1 keeps its course.")
    _assert("Department: Departamento de Informática" in y_block, "Department is included.")
    _assert("Retrieved content:" in y_block and "Texto do projecto Y." in y_block, "Content stays under S1.")
    _assert("Universidade Z" not in y_block, "S1 does not receive S2 metadata.")
    _assert(z_block.startswith("[S2]"), "Second block is S2.")
    _assert("Institution: Universidade Z" in z_block, "S2 keeps its institution.")
    _assert("Authors: Ana Lopes" in z_block and "Year: 2018" in z_block, "S2 metadata stays with S2.")
    _assert("Universidade Y" not in z_block, "S2 does not receive S1 metadata.")
    _assert("PDF page: 3" in y_block and "PDF page: 9" in z_block, "Physical pages stay with their labels.")


def test_question_values_do_not_replace_metadata() -> None:
    pack = EvidenceAssembler().assemble(
        [_ranked("Trecho recuperado.", "pfc-y", 1)],
        documents={
            "pfc-y": EvidenceDocument(
                pfc_id="pfc-y",
                title="Projecto Y",
                authors=("Maria Costa",),
                year=2021,
                institution="Universidade Y",
                course="Engenharia Informática",
            )
        },
    )
    context = pack.context_text
    _assert("Institution: Universidade Y" in context, "Institution comes from the catalogue.")
    _assert("Year: 2021" in context, "Year comes from the catalogue.")
    _assert("Authors: Maria Costa" in context, "Author comes from the catalogue.")
    _assert("Course: Engenharia Informática" in context, "Course comes from the catalogue.")
    for assumed in ("Universidade X", "2020", "João", "Medicina"):
        _assert(assumed not in context, f"Question-only value {assumed} must not enter metadata.")


def test_missing_optional_metadata_is_omitted() -> None:
    pack = EvidenceAssembler().assemble(
        [_ranked("Trecho sem curso.", "pfc-m", 2)],
        documents={
            "pfc-m": EvidenceDocument(
                pfc_id="pfc-m",
                title="Projecto incompleto",
                authors=("Maria Costa",),
                year=2021,
            )
        },
    )
    context = pack.context_text
    _assert("Title: Projecto incompleto" in context, "Present title is shown.")
    _assert("Institution:" not in context, "Missing institution is omitted.")
    _assert("Course:" not in context, "Missing course is omitted.")
    _assert("Department:" not in context, "Missing department is omitted.")
    _assert("None" not in context, "Missing fields are not rendered as None.")
    _assert("Trecho sem curso." in context, "Content is still supplied.")


def test_grounding_prompt_protects_metadata() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = MagicMock(content="Resposta [S1].")
    generator = AnswerGenerator(llm_client=llm, config=GenerationConfig())
    pack = EvidenceAssembler().assemble(
        [_ranked("Trecho.", "pfc-y", 1)],
        documents={
            "pfc-y": EvidenceDocument(
                pfc_id="pfc-y",
                title="Projecto Y",
                authors=("Maria Costa",),
                year=2021,
                institution="Universidade Y",
                course="Engenharia Informática",
            )
        },
    )
    generator.generate(
        "Que projectos da Universidade X de 2020 do autor João no curso de Medicina estão disponíveis?",
        pack,
    )
    system_prompt = llm.invoke_messages.call_args.args[0][0][1]
    user_prompt = llm.invoke_messages.call_args.args[0][1][1]
    for phrase in (
        "authoritative",
        "Never change a metadata value",
        "do not present non-matching documents",
        "retrieved evidence does not establish",
        "Do not infer institutional affiliation",
    ):
        _assert(phrase in system_prompt, f"Grounding rule missing: {phrase}")
    for forbidden in ("Universidade Eduardo Mondlane", "ISUTC", "Óscar", "Guibunda"):
        _assert(forbidden not in system_prompt, f"Prompt must not special-case {forbidden}.")
    _assert("Institution: Universidade Y" in user_prompt, "Authoritative institution reaches the generator.")
    _assert("Document metadata:" in user_prompt and "Retrieved content:" in user_prompt, "Metadata and content are separated.")
    _assert(
        "as evidências recuperadas não estabelecem" in user_prompt,
        "The answer-language instruction states conservative absence.",
    )
    _assert("cite cada afirmação factual" in user_prompt, "Citations stay required beside the metadata rule.")


def test_service_batches_lookup_and_reuses_it() -> None:
    catalogue = MagicMock()
    catalogue.find_by_ids.return_value = [
        _record(
            "pfc-y",
            title="Projecto Y",
            authors=("Maria Costa",),
            year=2021,
            institution="Universidade Y",
            course="Engenharia Informática",
        ),
        _record(
            "pfc-z",
            title="Projecto Z",
            authors=("Ana Lopes",),
            year=2018,
            institution="Universidade Z",
            course="Gestão",
        ),
    ]
    retriever = MagicMock(spec=HybridRetriever)
    reranker = MagicMock()
    generator = MagicMock()
    hits = [
        _ranked("Texto Y.", "pfc-y", 4),
        _ranked("Texto Z.", "pfc-z", 6),
    ]
    retriever.retrieve.return_value = [
        SearchResult(content=item.content, metadata=item.metadata, score=0.2)
        for item in hits
    ]
    reranker.rerank.return_value = hits
    generator.generate.return_value = "Apenas o segundo documento corresponde [S2]."
    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        evidence_assembler=EvidenceAssembler(),
        answer_generator=generator,
        citation_validator=CitationValidator(),
        catalogue=catalogue,
    )
    result = service.answer("Que projectos da Universidade X estão disponíveis?")
    _assert(catalogue.find_by_ids.call_count == 1, "Catalogue is queried once.")
    looked_up = catalogue.find_by_ids.call_args.args[0]
    _assert(looked_up == ["pfc-y", "pfc-z"], f"Distinct ids in one batch, got {looked_up}.")
    pack = generator.generate.call_args.args[1]
    _assert("Institution: Universidade Y" in pack.context_text, "S1 institution reaches generation.")
    _assert("Institution: Universidade Z" in pack.context_text, "S2 institution reaches generation.")
    _assert("Universidade X" not in pack.context_text, "The question institution is not written into evidence.")
    _assert(len(result.sources) == 1, "Only the cited source is returned.")
    _assert(result.sources[0].pfc_id == "pfc-z", "Citation maps to the cited PFC.")
    _assert(result.sources[0].title == "Projecto Z", "Cited source keeps catalogue title.")
    _assert(result.sources[0].authors == ("Ana Lopes",), "Cited source keeps catalogue author.")
    _assert(result.sources[0].year == 2018, "Cited source keeps catalogue year.")
    _assert(result.diagnostics.insufficient_evidence is False, "Supplied evidence is not flagged as absent.")

    invalid = CitationValidator().validate("Texto [S1] e [S9].", pack)
    _assert(invalid.invalid_citations == ("S9",), "Citation validation still rejects unknown labels.")
    _assert(invalid.sources[0].title is None, "The validator does not invent catalogue fields.")


def main() -> int:
    test_metadata_stays_with_its_citation_label()
    test_question_values_do_not_replace_metadata()
    test_missing_optional_metadata_is_omitted()
    test_grounding_prompt_protects_metadata()
    test_service_batches_lookup_and_reuses_it()
    print("All Slice 4C grounding checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
