"""Parse and validate inline citation labels against supplied evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.context.evidence_assembler import EvidenceItem, EvidencePack
from rag.service.models import SourceReference

# Accept [S1] and the common model variant (S1).
_CITATION_PATTERN = re.compile(r"[\[(](S\d+)[\])]", flags=re.IGNORECASE)


@dataclass(frozen=True)
class CitationValidationResult:
    cleaned_answer: str
    used_citation_ids: tuple[str, ...]
    sources: tuple[SourceReference, ...]
    invalid_citations: tuple[str, ...]


class CitationValidator:
    """Application-owned citation safety: labels must exist in the evidence pack."""

    def validate(self, answer: str, evidence: EvidencePack) -> CitationValidationResult:
        evidence_map = evidence.by_citation_id()
        valid_ids: list[str] = []
        invalid_ids: list[str] = []
        seen_valid: set[str] = set()
        seen_invalid: set[str] = set()

        for match in _CITATION_PATTERN.finditer(answer or ""):
            citation_id = self._normalize_id(match.group(1))
            if citation_id in evidence_map:
                if citation_id not in seen_valid:
                    seen_valid.add(citation_id)
                    valid_ids.append(citation_id)
            elif citation_id not in seen_invalid:
                seen_invalid.add(citation_id)
                invalid_ids.append(citation_id)

        cleaned = self._normalize_and_strip(answer or "", set(invalid_ids), set(valid_ids))
        sources = tuple(self._to_source(evidence_map[cid]) for cid in valid_ids)
        return CitationValidationResult(
            cleaned_answer=cleaned.strip(),
            used_citation_ids=tuple(valid_ids),
            sources=sources,
            invalid_citations=tuple(invalid_ids),
        )

    def _normalize_id(self, raw: str) -> str:
        raw = raw.strip()
        if re.fullmatch(r"[Ss]\d+", raw):
            return f"S{raw[1:]}"
        return raw

    def _normalize_and_strip(
        self,
        answer: str,
        invalid_ids: set[str],
        valid_ids: set[str],
    ) -> str:
        def _replace(match: re.Match[str]) -> str:
            citation_id = self._normalize_id(match.group(1))
            if citation_id in invalid_ids:
                return ""
            if citation_id in valid_ids:
                return f"[{citation_id}]"
            return match.group(0)

        cleaned = _CITATION_PATTERN.sub(_replace, answer)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" *\n{3,}", "\n\n", cleaned)
        return cleaned

    def _to_source(self, item: EvidenceItem) -> SourceReference:
        return SourceReference(
            citation_id=item.citation_id,
            source=item.source,
            page=item.page,
            chunk_index=item.chunk_index,
            chunk_id=item.chunk_id,
            excerpt=item.excerpt,
        )
