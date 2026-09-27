"""PFC catalogue listing and original PDF access."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from api.schemas import DocumentDetailResponse, DocumentSummaryResponse
from rag.documents.models import PfcRecord
from rag.documents.store import PfcStore

router = APIRouter(prefix="/api/v1", tags=["documents"])


def _store(request: Request) -> PfcStore:
    store = getattr(request.app.state, "pfc_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "detail": "The document catalogue is not ready.",
            },
        )
    return store


def _summary(record: PfcRecord) -> DocumentSummaryResponse:
    return DocumentSummaryResponse(
        id=record.id,
        title=record.title,
        authors=list(record.authors),
        year=record.year,
        course=record.course,
        status=record.status.value,
    )


def _detail(record: PfcRecord) -> DocumentDetailResponse:
    summary = _summary(record)
    return DocumentDetailResponse(
        **summary.model_dump(),
        institution=record.institution,
        department=record.department,
        supervisor=record.supervisor,
        abstract=record.abstract,
        keywords=list(record.keywords),
        language=record.language,
        original_filename=record.original_filename,
        page_count=record.page_count,
        error=record.error,
    )


@router.get("/documents", response_model=list[DocumentSummaryResponse])
def list_documents(
    request: Request,
    year: int | None = None,
    course: str | None = None,
) -> list[DocumentSummaryResponse]:
    records = _store(request).list_documents(year=year, course=course or None)
    return [_summary(record) for record in records]


@router.get("/documents/{pfc_id}", response_model=DocumentDetailResponse)
def get_document(pfc_id: str, request: Request) -> DocumentDetailResponse:
    record = _store(request).get(pfc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="PFC not found.")
    return _detail(record)


@router.get("/documents/{pfc_id}/file")
def get_document_file(pfc_id: str, request: Request) -> FileResponse:
    record = _store(request).get(pfc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="PFC not found.")
    corpus_dir = getattr(request.app.state, "pfc_corpus_directory", None)
    if not corpus_dir:
        raise HTTPException(status_code=404, detail="PFC file is not available.")
    path = _managed_pdf(record.storage_path, Path(corpus_dir))
    if path is None:
        raise HTTPException(status_code=404, detail="PFC file is not available.")
    filename = Path(record.original_filename).name or "document.pdf"
    return FileResponse(path, media_type="application/pdf", filename=filename)


def _managed_pdf(storage_path: str, corpus_dir: Path) -> Path | None:
    """Serve only files that live under the managed corpus directory."""
    path = Path(storage_path).resolve()
    root = corpus_dir.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path
