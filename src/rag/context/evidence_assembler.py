"""Turn ranked retrieval/rerank hits into labelled generation evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from rag.rerank.reranker import RerankResult


@dataclass(frozen=True)
class EvidenceDocument:
    """Authoritative catalogue fields for one PFC. Retrieval does not load these."""

    pfc_id: str
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    institution: str | None = None
    course: str | None = None
    department: str | None = None


@dataclass(frozen=True)
class EvidenceItem:
    citation_id: str
    content: str
    source: str
    page: int | str | None
    chunk_index: int | str | None
    chunk_id: str | None
    excerpt: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvidencePack:
    items: tuple[EvidenceItem, ...]
    context_text: str

    @property
    def is_empty(self) -> bool:
        return not self.items

    def by_citation_id(self) -> dict[str, EvidenceItem]:
        return {item.citation_id: item for item in self.items}


@dataclass(frozen=True)
class EvidenceAssemblerConfig:
    max_chunks: int = 5
    max_chars: int = 6000
    excerpt_chars: int = 280


class EvidenceAssemblerException(Exception):
    """Raised when evidence cannot be assembled."""


class EvidenceAssembler:
    """Selects bounded evidence and assigns deterministic [S1], [S2], ... labels."""

    def __init__(self, config: EvidenceAssemblerConfig | None = None):
        self.config = config or EvidenceAssemblerConfig()

    def assemble(
        self,
        ranked: list[RerankResult],
        documents: Mapping[str, EvidenceDocument] | None = None,
    ) -> EvidencePack:
        if self.config.max_chunks <= 0:
            raise EvidenceAssemblerException("max_chunks must be greater than zero.")
        if self.config.max_chars <= 0:
            raise EvidenceAssemblerException("max_chars must be greater than zero.")

        selected: list[EvidenceItem] = []
        used_keys: set[str] = set()
        used_chars = 0

        for result in ranked:
            if len(selected) >= self.config.max_chunks:
                break
            content = (result.content or "").strip()
            if not content:
                continue
            key = self._dedupe_key(result)
            if key in used_keys:
                continue

            metadata = dict(result.metadata or {})
            citation_id = f"S{len(selected) + 1}"
            document = _document_for(metadata, documents)
            header = _block_header(citation_id, metadata, document)
            separator = 2 if selected else 0
            remaining = self.config.max_chars - used_chars - separator - len(header) - 1
            if remaining <= 0:
                break
            clipped = content if len(content) <= remaining else content[:remaining].rstrip()
            if not clipped:
                break

            item = EvidenceItem(
                citation_id=citation_id,
                content=clipped,
                source=str(metadata.get("source", "unknown-source")),
                page=metadata.get("page"),
                chunk_index=metadata.get("chunk_index"),
                chunk_id=str(metadata["chunk_id"]) if metadata.get("chunk_id") else None,
                excerpt=self._excerpt(clipped),
                metadata=metadata,
            )
            selected.append(item)
            used_keys.add(key)
            used_chars += separator + len(header) + 1 + len(clipped)

        return EvidencePack(
            items=tuple(selected),
            context_text=self._format_context(selected, documents),
        )

    def _dedupe_key(self, result: RerankResult) -> str:
        metadata = result.metadata or {}
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        normalized = re.sub(r"\s+", " ", (result.content or "").strip().lower())
        return (
            f"{metadata.get('source')}:"
            f"{metadata.get('page')}:"
            f"{metadata.get('chunk_index')}:"
            f"{hash(normalized)}"
        )

    def _excerpt(self, content: str) -> str:
        cleaned = re.sub(r"\s+", " ", content).strip()
        limit = self.config.excerpt_chars
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip() + "..."

    def _format_context(
        self,
        items: list[EvidenceItem],
        documents: Mapping[str, EvidenceDocument] | None,
    ) -> str:
        if not items:
            return ""
        blocks: list[str] = []
        for item in items:
            document = _document_for(item.metadata, documents)
            header = _block_header(item.citation_id, item.metadata, document)
            blocks.append(f"{header}\n{item.content}")
        body = "\n\n".join(blocks)
        note = _institution_scope(items, documents)
        if not note:
            return body
        return f"{note}\n\n{body}"


def _document_for(
    metadata: dict[str, Any],
    documents: Mapping[str, EvidenceDocument] | None,
) -> EvidenceDocument | None:
    if not documents:
        return None
    raw_id = metadata.get("pfc_id")
    if not raw_id:
        return None
    return documents.get(str(raw_id))


def _institution_scope(
    items: list[EvidenceItem],
    documents: Mapping[str, EvidenceDocument] | None,
) -> str:
    """Name only the institutions the catalogue actually established for this evidence."""
    if not documents:
        return ""
    institutions: list[str] = []
    seen: set[str] = set()
    for item in items:
        document = _document_for(item.metadata, documents)
        institution = document.institution if document is not None else None
        if institution and institution not in seen:
            seen.add(institution)
            institutions.append(institution)
    if not institutions:
        return ""
    return (
        "Institutions established by the document metadata in this evidence: "
        + "; ".join(institutions)
        + ". Do not describe these sources as belonging to any other institution."
    )


def _metadata_lines(document: EvidenceDocument) -> list[str]:
    lines: list[str] = []
    if document.title:
        lines.append(f"Title: {document.title}")
    if document.authors:
        lines.append("Authors: " + "; ".join(document.authors))
    if document.year is not None:
        lines.append(f"Year: {document.year}")
    if document.institution:
        lines.append(f"Institution: {document.institution}")
    if document.course:
        lines.append(f"Course: {document.course}")
    if document.department:
        lines.append(f"Department: {document.department}")
    return lines


def _block_header(
    citation_id: str,
    metadata: dict[str, Any],
    document: EvidenceDocument | None,
) -> str:
    """Separate catalogue metadata from the retrieved passage."""
    lines = [f"[{citation_id}]"]
    meta_lines = _metadata_lines(document) if document is not None else []
    if meta_lines:
        lines.append("Document metadata:")
        lines.extend(meta_lines)
        lines.append(
            "These metadata values are authoritative for this source. "
            "Do not replace them with a different institution, author, year, or course."
        )
        page = metadata.get("page")
        lines.append("Retrieved content:")
        if page is not None:
            lines.append(f"PDF page: {page}")
        return "\n".join(lines)

    page = metadata.get("page")
    lines.append(f"Source: {metadata.get('source', 'unknown-source')}")
    lines.append(f"Page: {page if page is not None else 'unknown'}")
    lines.append("Retrieved content:")
    return "\n".join(lines)
