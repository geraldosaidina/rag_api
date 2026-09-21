"""Turn ranked retrieval/rerank hits into labelled generation evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag.rerank.reranker import RerankResult


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

    def assemble(self, ranked: list[RerankResult]) -> EvidencePack:
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

            block_overhead = 80
            remaining = self.config.max_chars - used_chars - block_overhead
            if remaining <= 0:
                break
            clipped = content if len(content) <= remaining else content[:remaining].rstrip()
            if not clipped:
                break

            citation_id = f"S{len(selected) + 1}"
            metadata = dict(result.metadata or {})
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
            used_chars += block_overhead + len(clipped)

        return EvidencePack(
            items=tuple(selected),
            context_text=self._format_context(selected),
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

    def _format_context(self, items: list[EvidenceItem]) -> str:
        if not items:
            return ""
        blocks: list[str] = []
        for item in items:
            blocks.append(
                "\n".join(
                    [
                        f"[{item.citation_id}]",
                        f"Source: {item.source}",
                        f"Page: {item.page if item.page is not None else 'unknown'}",
                        "Content:",
                        item.content,
                    ]
                )
            )
        return "\n\n".join(blocks)
