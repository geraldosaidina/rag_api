"""Register, index, replace, and delete PFC documents."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from langchain_core.documents import Document

from rag.documents.metadata import extract_pfc_metadata
from rag.documents.models import ExtractedPfcMetadata, PfcRecord, PfcStatus
from rag.documents.store import PfcStore
from rag.ingest.chunker import DocumentChunker
from rag.ingest.pdf_loader import PDFLoader, PDFLoaderException
from rag.vector.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

_MIN_WORDS = 40


class PfcIngestionException(Exception):
    """Raised when a PFC cannot be registered, replaced, or deleted."""


class PfcIngestionService:
    """Orchestrates catalogue state, managed files, and Chroma chunks."""

    def __init__(
        self,
        catalogue: PfcStore,
        vector_store: ChromaVectorStore,
        corpus_directory: str | Path,
        chunker: DocumentChunker | None = None,
    ):
        self.catalogue = catalogue
        self.vector_store = vector_store
        self.corpus_directory = Path(corpus_directory)
        self.chunker = chunker or DocumentChunker()

    def ingest_file(self, file_path: str | Path) -> PfcRecord:
        source = Path(file_path)
        data = source.read_bytes()
        content_hash = _sha256(data)
        existing = self.catalogue.find_by_hash(content_hash)
        if existing and existing.status != PfcStatus.FAILED:
            logger.info(
                "Duplicate PDF %s matches PFC %s; skipping re-index.",
                source.name,
                existing.id,
            )
            return existing
        if existing and existing.status == PfcStatus.FAILED:
            logger.info("Retrying failed PFC %s.", existing.id)
            return self._index(existing, source, data)

        pfc_id = str(uuid.uuid4())
        record = self.catalogue.register(
            pfc_id=pfc_id,
            original_filename=source.name,
            content_hash=content_hash,
            storage_path=str(self.corpus_directory / f"{pfc_id}.pdf"),
        )
        return self._index(record, source, data)

    def ingest_directory(self, directory: str | Path) -> list[PfcRecord]:
        paths = sorted(Path(directory).glob("*.pdf"))
        if not paths:
            raise PfcIngestionException(f"No PDF files found in {directory}")
        return [self.ingest_file(path) for path in paths]

    def replace(self, pfc_id: str, file_path: str | Path) -> PfcRecord:
        record = self.catalogue.get(pfc_id)
        if record is None:
            raise PfcIngestionException(f"Unknown PFC: {pfc_id}")
        source = Path(file_path)
        data = source.read_bytes()
        content_hash = _sha256(data)
        if content_hash == record.content_hash and record.status == PfcStatus.INDEXED:
            return record
        other = self.catalogue.find_by_hash(content_hash)
        if other is not None and other.id != record.id:
            raise PfcIngestionException(
                "PDF content already belongs to another PFC."
            )
        updated = self.catalogue.update(
            record.id,
            {
                "content_hash": content_hash,
                "original_filename": source.name,
            },
        )
        return self._index(updated, source, data)

    def delete(self, pfc_id: str) -> None:
        record = self.catalogue.get(pfc_id)
        if record is None:
            raise PfcIngestionException(f"Unknown PFC: {pfc_id}")
        self.vector_store.delete_by_pfc_id(pfc_id)
        path = Path(record.storage_path)
        if path.exists():
            path.unlink()
        self.catalogue.delete(pfc_id)

    def _index(self, record: PfcRecord, source: Path, data: bytes) -> PfcRecord:
        self.catalogue.update(
            record.id,
            {"status": PfcStatus.PROCESSING.value, "error": None},
        )
        try:
            self._write_managed_file(record, data)
            documents = self._load_documents(source)
            texts = [document.page_content or "" for document in documents]
            page_count = len(documents)
            if not _has_usable_text(texts):
                return self._fail(
                    record.id,
                    "No usable text extracted from the PDF.",
                    page_count=page_count,
                )
            metadata = extract_pfc_metadata(texts)
            missing = metadata.missing_required()
            if missing:
                return self._fail(
                    record.id,
                    "Required metadata missing: " + ", ".join(missing) + ".",
                    page_count=page_count,
                    metadata=metadata,
                )
            chunks = self._build_chunks(documents, record, metadata)
            if not chunks:
                return self._fail(
                    record.id,
                    "No chunks were created from the extracted text.",
                    page_count=page_count,
                    metadata=metadata,
                )
            self._delete_chunks(record.id, required=True)
            try:
                self.vector_store.add_documents(
                    documents=chunks,
                    ids=self.chunker.build_ids(chunks),
                )
            except Exception as exc:
                raise PfcIngestionException(
                    f"Indexing failed: {type(exc).__name__}: {exc}"
                ) from exc
            return self.catalogue.save_extraction(
                record.id,
                metadata,
                page_count=page_count,
                status=PfcStatus.INDEXED,
                error=None,
            )
        except PfcIngestionException as exc:
            logger.error("Indexing failed for PFC %s: %s", record.id, exc)
            return self._fail(record.id, str(exc))
        except PDFLoaderException:
            logger.exception("PDF extraction failed for PFC %s", record.id)
            return self._fail(record.id, "No usable text extracted from the PDF.")
        except Exception as exc:
            logger.exception("PFC ingestion failed for %s", record.id)
            return self._fail(
                record.id,
                f"Ingestion failed: {type(exc).__name__}: {exc}",
            )

    def _build_chunks(
        self,
        documents: list[Document],
        record: PfcRecord,
        metadata: ExtractedPfcMetadata,
    ) -> list[Document]:
        prepared: list[Document] = []
        for document in documents:
            chunk_metadata: dict[str, object] = {
                "source": record.original_filename,
                "page": document.metadata.get("page"),
                "pfc_id": record.id,
                "year": metadata.year,
            }
            if metadata.language:
                chunk_metadata["language"] = metadata.language
            prepared.append(
                Document(
                    page_content=document.page_content or "",
                    metadata=chunk_metadata,
                )
            )
        if not prepared:
            return []
        return self.chunker.chunk_documents(prepared)

    def _load_documents(self, source: Path) -> list[Document]:
        loader = PDFLoader(data_dir=str(source.parent))
        return loader.load_pdf(source, skip_page_errors=True)

    def _write_managed_file(self, record: PfcRecord, data: bytes) -> None:
        path = Path(record.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _fail(
        self,
        pfc_id: str,
        error: str,
        *,
        page_count: int | None = None,
        metadata: ExtractedPfcMetadata | None = None,
    ) -> PfcRecord:
        self._delete_chunks(pfc_id)
        message = error[:500]
        if metadata is None:
            changes: dict[str, object] = {
                "status": PfcStatus.FAILED.value,
                "error": message,
            }
            if page_count is not None:
                changes["page_count"] = page_count
            return self.catalogue.update(pfc_id, changes)
        return self.catalogue.save_extraction(
            pfc_id,
            metadata,
            page_count=page_count,
            status=PfcStatus.FAILED,
            error=message,
        )

    def _delete_chunks(self, pfc_id: str, *, required: bool = False) -> None:
        try:
            self.vector_store.delete_by_pfc_id(pfc_id)
        except Exception as exc:
            logger.exception("Failed to delete Chroma chunks for PFC %s", pfc_id)
            if required:
                raise PfcIngestionException(
                    f"Indexing failed: could not delete existing chunks for {pfc_id}"
                ) from exc


def recover_processing_documents(
    catalogue: PfcStore,
    vector_store: ChromaVectorStore,
) -> list[str]:
    """Mark interrupted ingestions as failed and drop their chunks."""
    recovered: list[str] = []
    for record in catalogue.find_by_status(PfcStatus.PROCESSING):
        try:
            vector_store.delete_by_pfc_id(record.id)
        except Exception:
            logger.exception(
                "Could not delete chunks for interrupted PFC %s", record.id
            )
        catalogue.update(
            record.id,
            {
                "status": PfcStatus.FAILED.value,
                "error": "Ingestion was interrupted and marked failed on startup.",
            },
        )
        recovered.append(record.id)
        logger.info("Recovered interrupted PFC %s as failed.", record.id)
    return recovered


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _has_usable_text(pages: list[str]) -> bool:
    return sum(len(page.split()) for page in pages) >= _MIN_WORDS
