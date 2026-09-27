"""PFC catalogue records. SQLite is the authority for this data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PfcStatus(str, Enum):
    REGISTERED = "registered"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True)
class PfcRecord:
    """One PFC catalogue row."""

    id: str
    title: str | None
    authors: tuple[str, ...]
    year: int | None
    course: str | None
    institution: str | None
    department: str | None
    supervisor: str | None
    abstract: str | None
    keywords: tuple[str, ...]
    language: str | None
    original_filename: str
    content_hash: str
    storage_path: str
    page_count: int | None
    status: PfcStatus
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ExtractedPfcMetadata:
    """Deterministic metadata parsed from PDF text. Missing values stay empty."""

    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    course: str | None = None
    institution: str | None = None
    department: str | None = None
    supervisor: str | None = None
    abstract: str | None = None
    keywords: tuple[str, ...] = ()
    language: str | None = None
    warnings: tuple[str, ...] = ()

    def missing_required(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.title:
            missing.append("title")
        if not self.authors:
            missing.append("author")
        if self.year is None:
            missing.append("year")
        return tuple(missing)
