"""SQLite catalogue for PFC identity, metadata, and ingestion status."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from rag.documents.models import ExtractedPfcMetadata, PfcRecord, PfcStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pfc_documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    course TEXT,
    institution TEXT,
    department TEXT,
    supervisor TEXT,
    abstract TEXT,
    keywords TEXT NOT NULL DEFAULT '[]',
    language TEXT,
    original_filename TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    page_count INTEGER,
    status TEXT NOT NULL CHECK (
        status IN ('registered', 'processing', 'indexed', 'failed')
    ),
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_COLUMNS = (
    "title",
    "authors",
    "year",
    "course",
    "institution",
    "department",
    "supervisor",
    "abstract",
    "keywords",
    "language",
    "original_filename",
    "content_hash",
    "storage_path",
    "page_count",
    "status",
    "error",
)


class PfcStoreException(Exception):
    """Raised when the PFC catalogue cannot complete an operation."""


class PfcStore:
    """Small sqlite3 catalogue. Not a generic repository."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def register(
        self,
        *,
        pfc_id: str,
        original_filename: str,
        content_hash: str,
        storage_path: str,
    ) -> PfcRecord:
        now = _now()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO pfc_documents (
                        id, authors, keywords, original_filename, content_hash,
                        storage_path, status, created_at, updated_at
                    ) VALUES (?, '[]', '[]', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pfc_id,
                        original_filename,
                        content_hash,
                        storage_path,
                        PfcStatus.REGISTERED.value,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise PfcStoreException(
                f"Could not register PFC {pfc_id}: {exc}"
            ) from exc
        return self._require(pfc_id)

    def get(self, pfc_id: str) -> PfcRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pfc_documents WHERE id = ?",
                (pfc_id,),
            ).fetchone()
        return _record(row) if row else None

    def find_by_hash(self, content_hash: str) -> PfcRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pfc_documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return _record(row) if row else None

    def find_by_ids(self, pfc_ids: list[str]) -> list[PfcRecord]:
        if not pfc_ids:
            return []
        placeholders = ",".join("?" for _ in pfc_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pfc_documents WHERE id IN ({placeholders})",
                tuple(pfc_ids),
            ).fetchall()
        return [_record(row) for row in rows]

    def find_by_status(self, status: PfcStatus) -> list[PfcRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pfc_documents WHERE status = ? ORDER BY updated_at",
                (status.value,),
            ).fetchall()
        return [_record(row) for row in rows]

    def list_documents(
        self,
        *,
        year: int | None = None,
        course: str | None = None,
    ) -> list[PfcRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if year is not None:
            clauses.append("year = ?")
            params.append(year)
        if course:
            clauses.append("course = ?")
            params.append(course)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM pfc_documents
                {where}
                ORDER BY year IS NULL, year DESC, title
                """,
                tuple(params),
            ).fetchall()
        return [_record(row) for row in rows]

    def update(self, pfc_id: str, changes: dict[str, object]) -> PfcRecord:
        unknown = set(changes) - set(_COLUMNS)
        if unknown:
            raise PfcStoreException(f"Unsupported PFC columns: {sorted(unknown)}")
        if not changes:
            return self._require(pfc_id)
        assignments = ", ".join(f"{column} = ?" for column in changes)
        values = [_adapt(column, value) for column, value in changes.items()]
        values.append(_now())
        values.append(pfc_id)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE pfc_documents SET {assignments}, updated_at = ? WHERE id = ?",
                tuple(values),
            )
            if cursor.rowcount != 1:
                raise PfcStoreException(f"PFC not found: {pfc_id}")
        return self._require(pfc_id)

    def save_extraction(
        self,
        pfc_id: str,
        metadata: ExtractedPfcMetadata,
        *,
        page_count: int | None,
        status: PfcStatus,
        error: str | None,
    ) -> PfcRecord:
        return self.update(
            pfc_id,
            {
                "title": metadata.title,
                "authors": list(metadata.authors),
                "year": metadata.year,
                "course": metadata.course,
                "institution": metadata.institution,
                "department": metadata.department,
                "supervisor": metadata.supervisor,
                "abstract": metadata.abstract,
                "keywords": list(metadata.keywords),
                "language": metadata.language,
                "page_count": page_count,
                "status": status.value,
                "error": error,
            },
        )

    def delete(self, pfc_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM pfc_documents WHERE id = ?", (pfc_id,))
            if cursor.rowcount != 1:
                raise PfcStoreException(f"PFC not found: {pfc_id}")

    def _require(self, pfc_id: str) -> PfcRecord:
        record = self.get(pfc_id)
        if record is None:
            raise PfcStoreException(f"PFC not found: {pfc_id}")
        return record

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _adapt(column: str, value: object) -> object:
    if column in {"authors", "keywords"}:
        return json.dumps(list(value or []), ensure_ascii=False)
    if isinstance(value, PfcStatus):
        return value.value
    return value


def _record(row: sqlite3.Row) -> PfcRecord:
    return PfcRecord(
        id=row["id"],
        title=row["title"],
        authors=tuple(json.loads(row["authors"] or "[]")),
        year=row["year"],
        course=row["course"],
        institution=row["institution"],
        department=row["department"],
        supervisor=row["supervisor"],
        abstract=row["abstract"],
        keywords=tuple(json.loads(row["keywords"] or "[]")),
        language=row["language"],
        original_filename=row["original_filename"],
        content_hash=row["content_hash"],
        storage_path=row["storage_path"],
        page_count=row["page_count"],
        status=PfcStatus(row["status"]),
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
