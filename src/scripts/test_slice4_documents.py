"""Deterministic Slice 4B tests for catalogue, ingestion, citations, and document API.

Run:
  uv run python src/scripts/test_slice4_documents.py
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from api.app import create_app
from rag.citations.citation_validator import CitationValidator
from rag.context.evidence_assembler import EvidenceAssembler
from rag.documents.front_matter import extract_abstract, extract_keywords
from rag.documents.ingestion import (
    PfcIngestionException,
    PfcIngestionService,
    recover_processing_documents,
)
from rag.documents.metadata import extract_pfc_metadata
from rag.documents.models import PfcRecord, PfcStatus
from rag.documents.store import PfcStore
from rag.generation.answer_generator import AnswerGenerator
from rag.ingest.chunker import DocumentChunker
from rag.ingest.pdf_loader import PDFLoader
from rag.rerank.reranker import RerankResult
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.service.rag_service import RAGService
from rag.vector.chroma_store import ChromaConfig, ChromaVectorStore, SearchResult


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, float((index % 5) + 1) / 10] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        _ = text
        return [0.1, 0.2, 0.3]


def _write_pdf(path: Path, text: str) -> None:
    lines = text.splitlines() or [""]
    commands = ["BT /F1 11 Tf 48 760 Td"]
    for index, line in enumerate(lines):
        escaped = (
            line.encode("latin-1", errors="replace")
            .decode("latin-1")
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        if index:
            commands.append("0 -13 Td")
        commands.append(f"({escaped}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        ),
        (
            b"4 0 obj << /Length "
            + str(len(stream)).encode("ascii")
            + b" >> stream\n"
            + stream
            + b"\nendstream endobj\n"
        ),
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(output))
        output.extend(item)
    xref = len(output)
    output.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(output)


def _cover(
    *,
    title: str = "Sistema de teste para ingestao documental",
    author: str = "Ana Maria Silva",
    course: str = "Licenciatura em Engenharia Informatica e de Telecomunicacoes",
    year_line: str = "Maio de 2019",
    supervisor: str = "Supervisor:\nEng. Carlos Mendes",
    department: str = "Departamento de Tecnologias de Informacao e Comunicacao",
    extra: str = "",
) -> str:
    return "\n".join(
        [
            "INSTITUTO SUPERIOR DE TRANSPORTES E COMUNICACOES",
            "",
            title,
            "",
            author,
            "",
            "Projecto Final do Curso",
            course,
            "",
            supervisor,
            "",
            department,
            "",
            year_line,
            "",
            extra,
        ]
    )


def _resumo(marker: str = "ALPHA-MARKER") -> str:
    body = " ".join(["Este projecto de teste descreve a ingestao documental."] * 8)
    return f"RESUMO\n{body} O marcador e {marker}.\nPALAVRAS-CHAVE: teste, ingestao\n"


def _service(root: Path, vectors) -> PfcIngestionService:
    store = PfcStore(root / "pfc.db")
    store.initialize()
    corpus = root / "corpus"
    corpus.mkdir()
    return PfcIngestionService(store, vectors, corpus)


def test_metadata_fields_and_formats() -> None:
    normal = extract_pfc_metadata(
        [
            _cover(),
            _resumo(),
        ]
    )
    _assert(normal.title == "Sistema de teste para ingestao documental", "Title.")
    _assert(normal.authors == ("Ana Maria Silva",), "Author list.")
    _assert(normal.year == 2019, "Year Maio de 2019.")
    _assert("Licenciatura" in (normal.course or ""), "Course.")
    _assert(normal.institution and "INSTITUTO" in normal.institution.upper(), "Institution.")
    _assert(normal.department and normal.department.startswith("Departamento"), "Department.")
    _assert(normal.supervisor and "Carlos Mendes" in normal.supervisor, "Labelled supervisor.")
    _assert(normal.keywords == ("teste", "ingestao"), "Keywords.")
    _assert(normal.language == "pt", "Portuguese language.")
    _assert(normal.abstract and "PALAVRAS" not in normal.abstract, "Resumo stops before keywords.")
    _assert(not normal.missing_required(), "Required metadata present.")

    flattened = (
        "INSTITUTO SUPERIOR DE TRANSPORTES E COMUNICACOES    "
        "AVALIACAO DE VULNERABILIDADES EM PORTAIS    "
        "Vagner Flavio Fafetine Nhachungue    "
        "Projecto Final do Curso Licenciatura em Engenharia Informatica e de Telecomunicacoes "
        "Eng. Emircio Zeca Vieira    "
        "Departamento de Tecnologias de Informacao e Comunicacao    "
        "Junho, 2024"
    )
    flat = extract_pfc_metadata([flattened, _resumo()])
    _assert(flat.authors == ("Vagner Flavio Fafetine Nhachungue",), "Flattened author.")
    _assert(flat.year == 2024, "Flattened year.")
    _assert("AVALIACAO" in (flat.title or ""), "Flattened title.")
    _assert("Eng." not in (flat.course or ""), "Course excludes the supervisor name.")

    august = extract_pfc_metadata([_cover(year_line="Agosto,2011")])
    _assert(august.year == 2011, "Agosto,2011.")
    june = extract_pfc_metadata([_cover(year_line="Junho, 2016")])
    _assert(june.year == 2016, "Junho, 2016.")

    dtic = extract_pfc_metadata([_cover(department="DTIC")])
    _assert(dtic.department == "DTIC", "DTIC department.")

    ack = extract_pfc_metadata(
        [
            _cover(supervisor=""),
            "Agradecimentos. Ao meu supervisor, Eng. Carlos Mendes, pela orientacao neste trabalho.",
        ]
    )
    _assert(ack.supervisor and "Carlos Mendes" in ack.supervisor, "Acknowledgement supervisor.")
    _assert("supervisor_from_acknowledgement" in ack.warnings, "Acknowledgement warning.")

    for label in (
        "PALAVRAS CHAVE: redes, acesso",
        "PALAVRAS-CHAVE: redes, acesso",
        "PALAVRAS – CHAVE: redes, acesso",
        "PALAVRAS - CHAVE: redes, acesso",
        "PALAVRAS CHAVES: redes, acesso",
    ):
        keywords = extract_keywords([label])
        _assert(keywords == ("redes", "acesso"), f"Keyword label failed: {label}")
    _assert(extract_keywords(["Sem lista de termos."]) == (), "Missing keywords.")
    abstract = extract_abstract(
        ["RESUMO\n" + ("Texto de resumo suficiente. " * 12) + "\nPALAVRAS-CHAVE: a"]
    )
    _assert(abstract and "PALAVRAS" not in abstract and len(abstract) >= 80, "Resumo boundary.")

    incomplete = extract_pfc_metadata(["Apenas um texto sem capa."])
    _assert(incomplete.missing_required() == ("title", "author", "year"), "Missing required.")


def test_reference_headings_and_chunk_identity() -> None:
    chunker = DocumentChunker()
    references = chunker.chunk_documents(
        [
            Document(
                page_content="Referências\nSilva, Ana. Um titulo academico publicado em 2019.",
                metadata={"source": "old.pdf", "page": 12, "pfc_id": "pfc-1", "year": 2019},
            )
        ]
    )
    bibliography = chunker.chunk_documents(
        [
            Document(
                page_content="Bibliografia\nMondlane, Arlindo. Outro titulo publicado em 2018.",
                metadata={"source": "old.pdf", "page": 13, "pfc_id": "pfc-1"},
            )
        ]
    )
    _assert(references[0].metadata["section_type"] == "references", "Referências.")
    _assert(references[0].metadata["retrieval_quality"] == "low", "Referências quality.")
    _assert(bibliography[0].metadata["section_type"] == "references", "Bibliografia.")
    _assert(bibliography[0].metadata["retrieval_quality"] == "low", "Bibliografia quality.")
    _assert(
        str(references[0].metadata["chunk_id"]).startswith("pfc-1:page-12:chunk-0:"),
        "Chunk id uses pfc_id.",
    )
    _assert(references[0].metadata["year"] == 2019, "Year stays on the chunk.")


def test_duplicate_retry_and_failed_extraction() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        vectors = MagicMock()
        service = _service(root, vectors)
        pdf = root / "ana.pdf"
        _write_pdf(pdf, _cover() + "\n" + _resumo())
        first = service.ingest_file(pdf)
        second = service.ingest_file(pdf)
        renamed = root / "outro-nome.pdf"
        renamed.write_bytes(pdf.read_bytes())
        third = service.ingest_file(renamed)
        _assert(uuid.UUID(first.id), "UUID identity.")
        _assert(len(first.content_hash) == 64, "SHA-256 identity.")
        _assert(first.id == second.id == third.id, "Same bytes share one PFC.")
        _assert(first.original_filename == "ana.pdf", "Filename is not replaced by a duplicate.")
        _assert(third.original_filename == "ana.pdf", "Duplicate filename is ignored.")
        _assert(vectors.add_documents.call_count == 1, "Duplicate content is not embedded again.")
        _assert((root / "corpus" / f"{first.id}.pdf").is_file(), "Managed copy.")

        blank = root / "blank.pdf"
        _write_pdf(blank, "curto")
        failed = service.ingest_file(blank)
        retried = service.ingest_file(blank)
        rows = service.catalogue.list_documents()
        _assert(failed.status == PfcStatus.FAILED, "Empty text fails.")
        _assert("No usable text" in (failed.error or ""), "Extraction error.")
        _assert(failed.id == retried.id, "Failed hash retries the same PFC.")
        _assert(sum(1 for row in rows if row.id == failed.id) == 1, "No duplicate failed row.")
        _assert(vectors.add_documents.call_count == 1, "Failed extraction writes no chunks.")

        missing_year = root / "no-year.pdf"
        _write_pdf(missing_year, _cover(year_line="") + "\n" + _resumo())
        incomplete = service.ingest_file(missing_year)
        _assert(incomplete.status == PfcStatus.FAILED, "Missing year fails.")
        _assert("year" in (incomplete.error or ""), "Missing-year diagnostic.")


def test_indexing_failure_is_compensated() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        vectors = MagicMock()
        vectors.add_documents.side_effect = RuntimeError("chroma down")
        service = _service(root, vectors)
        pdf = root / "ana.pdf"
        _write_pdf(pdf, _cover() + "\n" + _resumo())
        record = service.ingest_file(pdf)
        _assert(record.status == PfcStatus.FAILED, "Indexing failure is failed.")
        _assert(record.status != PfcStatus.INDEXED, "Must not stay indexed.")
        _assert("Indexing failed" in (record.error or ""), "Indexing diagnostic.")
        _assert(vectors.delete_by_pfc_id.called, "Chunks are deleted after failure.")


def test_real_chroma_replace_delete_restart_and_recovery() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        store = PfcStore(root / "pfc.db")
        store.initialize()
        corpus = root / "corpus"
        corpus.mkdir()
        vectors = ChromaVectorStore(
            embedding_function=_FakeEmbeddings(),
            config=ChromaConfig(
                persist_directory=str(root / "chroma"),
                collection_name="pfc_test",
            ),
        )
        service = PfcIngestionService(store, vectors, corpus)
        alpha = root / "alpha.pdf"
        beta = root / "beta.pdf"
        _write_pdf(alpha, _cover(extra=_resumo("ALPHA-MARKER")))
        _write_pdf(
            beta,
            _cover(
                title="Sistema de videovigilancia de teste",
                author="Belso Bento Langa",
                year_line="Junho, 2016",
                extra=_resumo("BETA-MARKER"),
            ),
        )
        indexed = service.ingest_file(alpha)
        _assert(indexed.status == PfcStatus.INDEXED, "Successful ingest is indexed.")
        chunk_count = vectors.count_where({"pfc_id": indexed.id})
        _assert(chunk_count > 0, "Chunks exist.")
        sample = vectors._client.get(where={"pfc_id": indexed.id}, include=["metadatas"])
        meta = sample["metadatas"][0]
        _assert(meta.get("pfc_id") == indexed.id, "pfc_id on chunk.")
        _assert(meta.get("year") == 2019, "Year copied to Chroma.")
        _assert(meta.get("language") == "pt", "Language copied to Chroma.")
        _assert(str(meta.get("chunk_id", "")).startswith(indexed.id), "Chunk id prefix.")

        reopened = PfcStore(root / "pfc.db")
        persisted = reopened.get(indexed.id)
        _assert(persisted is not None and persisted.status == PfcStatus.INDEXED, "Restart.")
        _assert(persisted.title == indexed.title, "Metadata survives restart.")

        replaced = service.replace(indexed.id, beta)
        _assert(replaced.id == indexed.id, "Replace keeps pfc_id.")
        _assert(replaced.content_hash != indexed.content_hash, "New hash.")
        _assert(replaced.year == 2016, "Replaced year.")
        _assert(replaced.original_filename == "beta.pdf", "Explicit replace updates filename.")
        documents = vectors._client.get(
            where={"pfc_id": indexed.id}, include=["documents"]
        )["documents"]
        blob = " ".join(documents)
        _assert("BETA-MARKER" in blob, "New text is indexed.")
        _assert("ALPHA-MARKER" not in blob, "Old vectors were removed.")

        reopened.update(
            indexed.id,
            {"status": PfcStatus.PROCESSING.value, "error": None},
        )
        recovered = recover_processing_documents(reopened, vectors)
        _assert(recovered == [indexed.id], "Startup finds the processing row.")
        failed = reopened.get(indexed.id)
        _assert(failed is not None and failed.status == PfcStatus.FAILED, "Recovered as failed.")
        _assert(vectors.count_where({"pfc_id": indexed.id}) == 0, "Stale chunks removed.")

        restored = service.replace(indexed.id, beta)
        _assert(restored.status == PfcStatus.INDEXED, "Replace after recovery reindexes.")
        service.delete(indexed.id)
        _assert(store.get(indexed.id) is None, "Catalogue row deleted.")
        _assert(not (corpus / f"{indexed.id}.pdf").exists(), "Managed file deleted.")
        _assert(vectors.count_where({"pfc_id": indexed.id}) == 0, "Vectors deleted.")


def test_replace_rejects_another_pfcs_bytes() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        service = _service(root, MagicMock())
        first_pdf = root / "a.pdf"
        second_pdf = root / "b.pdf"
        _write_pdf(first_pdf, _cover(extra=_resumo("ONE")))
        _write_pdf(second_pdf, _cover(author="Outra Pessoa Nome", extra=_resumo("TWO")))
        first = service.ingest_file(first_pdf)
        service.ingest_file(second_pdf)
        try:
            service.replace(first.id, second_pdf)
            raise AssertionError("Conflicting hash must be rejected.")
        except PfcIngestionException:
            pass
        current = service.catalogue.get(first.id)
        _assert(current is not None and current.status == PfcStatus.INDEXED, "Original stays indexed.")
        _assert(current.content_hash == first.content_hash, "Hash unchanged after rejected replace.")


def test_citation_enrichment_is_batched() -> None:
    catalogue = MagicMock()
    catalogue.find_by_ids.return_value = [
        PfcRecord(
            id="pfc-a",
            title="Sistema de teste",
            authors=("Ana Maria Silva",),
            year=2019,
            course=None,
            institution=None,
            department=None,
            supervisor=None,
            abstract=None,
            keywords=(),
            language="pt",
            original_filename="ana.pdf",
            content_hash="abc",
            storage_path="data/corpus/pfc-a.pdf",
            page_count=1,
            status=PfcStatus.INDEXED,
            error=None,
            created_at="",
            updated_at="",
        )
    ]
    retriever = MagicMock(spec=HybridRetriever)
    reranker = MagicMock()
    generator = MagicMock(spec=AnswerGenerator)
    retriever.retrieve.return_value = [
        SearchResult(
            content="Evidencia sobre ingestao.",
            metadata={
                "source": "ana.pdf",
                "page": 4,
                "chunk_index": 0,
                "chunk_id": "pfc-a:page-4:chunk-0:deadbeef",
                "pfc_id": "pfc-a",
            },
            score=0.2,
        ),
        SearchResult(
            content="Segunda evidencia do mesmo PFC.",
            metadata={
                "source": "ana.pdf",
                "page": 5,
                "chunk_index": 1,
                "chunk_id": "pfc-a:page-5:chunk-1:cafebabe",
                "pfc_id": "pfc-a",
            },
            score=0.3,
        ),
    ]
    reranker.rerank.return_value = [
        RerankResult(
            content=item.content,
            metadata=item.metadata,
            original_score=item.score,
            rerank_score=0.9,
            answerability_score=0.0,
            final_score=0.9,
        )
        for item in retriever.retrieve.return_value
    ]
    generator.generate.return_value = "A ingestao funciona [S1] e continua [S2]."
    service = RAGService(
        retriever=retriever,
        reranker=reranker,
        evidence_assembler=EvidenceAssembler(),
        answer_generator=generator,
        citation_validator=CitationValidator(),
        catalogue=catalogue,
    )
    result = service.answer("Como funciona a ingestao?")
    _assert(len(result.sources) == 2, "Both citations survive validation.")
    _assert(catalogue.find_by_ids.call_count == 1, "One catalogue lookup.")
    _assert(catalogue.find_by_ids.call_args.args[0] == ["pfc-a"], "Distinct pfc ids.")
    _assert(result.sources[0].title == "Sistema de teste", "Title enriched.")
    _assert(result.sources[0].authors == ("Ana Maria Silva",), "Authors enriched.")
    _assert(result.sources[0].year == 2019, "Year enriched.")
    _assert(result.sources[0].page == 4, "Physical page preserved.")
    _assert(result.sources[1].pfc_id == "pfc-a", "Second source shares the PFC.")
    invalid = CitationValidator().validate(
        "Texto [S1] e invalido [S9].",
        EvidenceAssembler().assemble(reranker.rerank.return_value[:1]),
    )
    _assert(invalid.invalid_citations == ("S9",), "Invalid labels are still rejected.")
    _assert("[S9]" not in invalid.cleaned_answer, "Invalid labels are stripped.")


def test_document_api() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        corpus = root / "corpus"
        corpus.mkdir()
        store = PfcStore(root / "pfc.db")
        store.initialize()
        pdf = corpus / "pfc-1.pdf"
        _write_pdf(pdf, "conteudo")
        outside = root / "secret.pdf"
        outside.write_bytes(b"%PDF-1.4 secret")
        store.register(
            pfc_id="pfc-1",
            original_filename="ana.pdf",
            content_hash="a" * 64,
            storage_path=str(pdf),
        )
        store.save_extraction(
            "pfc-1",
            extract_pfc_metadata([_cover(extra=_resumo())]),
            page_count=1,
            status=PfcStatus.INDEXED,
            error=None,
        )
        store.register(
            pfc_id="pfc-2",
            original_filename="outside.pdf",
            content_hash="b" * 64,
            storage_path=str(outside),
        )
        app = create_app(
            rag_service=MagicMock(),
            pfc_store=store,
            pfc_corpus_directory=str(corpus),
            initialize_rag=False,
        )
        client = TestClient(app)
        listing = client.get("/api/v1/documents")
        _assert(listing.status_code == 200, "Catalogue list.")
        body = listing.json()
        _assert(any(item["id"] == "pfc-1" and item["status"] == "indexed" for item in body), "Summary.")
        _assert("abstract" not in body[0], "List stays lightweight.")
        filtered = client.get("/api/v1/documents", params={"year": 2019})
        _assert(len(filtered.json()) == 1, "Year filter.")
        detail = client.get("/api/v1/documents/pfc-1")
        _assert(detail.status_code == 200, "Detail route.")
        _assert(detail.json()["authors"] == ["Ana Maria Silva"], "Detail authors.")
        _assert("storage_path" not in detail.json(), "Filesystem path stays internal.")
        downloaded = client.get("/api/v1/documents/pfc-1/file")
        _assert(downloaded.status_code == 200, "File route.")
        _assert(downloaded.headers["content-type"].startswith("application/pdf"), "PDF media type.")
        _assert(downloaded.content.startswith(b"%PDF"), "PDF bytes.")
        missing = client.get("/api/v1/documents/does-not-exist/file")
        _assert(missing.status_code == 404, "Unknown PFC is 404.")
        escaped = client.get("/api/v1/documents/..%2F..%2Fsecret/file")
        _assert(escaped.status_code == 404, "Path-like id is not a file lookup.")
        blocked = client.get("/api/v1/documents/pfc-2/file")
        _assert(blocked.status_code == 404, "Path outside the corpus is refused.")


def test_real_sample_metadata() -> None:
    samples = PROJECT_ROOT / "data" / "pfc_samples"
    _assert(samples.is_dir(), "Sample directory is required.")
    expected_years = {
        "Nereid": 2019,
        "BELSO": 2016,
        "Edson": 2018,
        "Mércia": 2018,
        "Hassan": 2025,
        "Lect": 2024,
        "Mayra": 2015,
        "Oscar": 2019,
        "Vagner": 2024,
        "PFC.pdf": 2025,
        "Arnaldo": 2021,
        "Diana": 2015,
        "Felipe": 2011,
        "Eugenia": 2023,
        "Nhacuonga": 2023,
    }
    loader = PDFLoader(str(samples))
    readable = 0
    for path in sorted(samples.glob("*.pdf")):
        documents = loader.load_pdf(path, skip_page_errors=True)
        pages = [document.page_content or "" for document in documents]
        metadata = extract_pfc_metadata(pages)
        if "Jaime" in path.name or "Mulhovo" in path.name:
            _assert(metadata.missing_required() == ("title", "author", "year"), path.name)
            _assert(sum(len(page.split()) for page in pages) < 40, "Jaime has no usable text.")
            continue
        readable += 1
        _assert(not metadata.missing_required(), f"Required metadata for {path.name}")
        _assert(metadata.language == "pt", f"Language for {path.name}")
        _assert(metadata.institution and "INSTITUTO" in metadata.institution.upper(), path.name)
        _assert(metadata.course and "Licenciatura" in metadata.course, path.name)
        year = next(value for key, value in expected_years.items() if key in path.name)
        _assert(metadata.year == year, f"Year for {path.name}: {metadata.year}")
        if "BELSO" in path.name:
            _assert(metadata.keywords == (), "Belso has no keywords.")
        else:
            _assert(metadata.keywords, f"Keywords for {path.name}")
        if "Arnaldo" in path.name:
            _assert(metadata.department and metadata.department.upper() == "DTIC", "DTIC.")
    _assert(readable == 14, f"Expected 14 readable samples, got {readable}.")


def main() -> int:
    test_metadata_fields_and_formats()
    test_reference_headings_and_chunk_identity()
    test_duplicate_retry_and_failed_extraction()
    test_indexing_failure_is_compensated()
    test_real_chroma_replace_delete_restart_and_recovery()
    test_replace_rejects_another_pfcs_bytes()
    test_citation_enrichment_is_batched()
    test_document_api()
    test_real_sample_metadata()
    print("All Slice 4 document checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
