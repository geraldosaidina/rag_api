"""Automatic citation and abstention helpers for Phase 2 (H1/H4).

These scores are structural / gold-aligned checks. They do not replace
semantic faithfulness judging against the supplied evidence pack.
``answer_notes`` from the gold set must not be treated as canonical answers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _as_int_page(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def score_citations(
    *,
    sources: Sequence[Mapping[str, Any]],
    primary_pfc_ids: Sequence[str],
    gold_pages_by_pfc: Mapping[str, Sequence[int]],
    unsupported: bool,
    invalid_citations_removed: Sequence[str],
) -> dict[str, Any]:
    """Document/page citation accuracy relative to gold primary evidence."""
    primary = {str(item) for item in primary_pfc_ids}
    pages_by_pfc = {
        str(pfc_id): {int(page) for page in pages}
        for pfc_id, pages in gold_pages_by_pfc.items()
    }

    cited_docs: list[str] = []
    seen_docs: set[str] = set()
    doc_hits = 0
    page_evaluable = 0
    page_hits = 0
    missing_pfc_id = 0

    for source in sources:
        pfc_id = source.get("pfc_id")
        if not pfc_id:
            missing_pfc_id += 1
            continue
        pfc_id = str(pfc_id)
        if pfc_id not in seen_docs:
            seen_docs.add(pfc_id)
            cited_docs.append(pfc_id)
            if not unsupported and pfc_id in primary:
                doc_hits += 1
            if unsupported and pfc_id in primary:
                # Gold has empty primary for unsupported; keep branch explicit.
                doc_hits += 1

        page = _as_int_page(source.get("page"))
        expected_pages = pages_by_pfc.get(pfc_id)
        if expected_pages is not None and page is not None:
            page_evaluable += 1
            if page in expected_pages:
                page_hits += 1

    cited_count = len(cited_docs)
    return {
        "cited_pfc_ids": cited_docs,
        "citation_count": len(list(sources)),
        "distinct_cited_documents": cited_count,
        "sources_missing_pfc_id": missing_pfc_id,
        "invalid_citations_removed": list(invalid_citations_removed),
        "invalid_citation_count": len(list(invalid_citations_removed)),
        "citation_doc_precision": (
            None
            if unsupported or cited_count == 0
            else doc_hits / float(cited_count)
        ),
        "citation_doc_recall": (
            None
            if unsupported or not primary
            else len(seen_docs & primary) / float(len(primary))
        ),
        "citation_page_precision": (
            None
            if page_evaluable == 0
            else page_hits / float(page_evaluable)
        ),
        "citation_page_evaluable_count": page_evaluable,
        "citation_page_hit_count": page_hits,
        # For unsupported questions, citing documents is not automatically wrong
        # (retrieved context may still be attached), but presenting them as
        # matching the false premise is judged separately.
        "unsupported_has_sources": bool(unsupported and cited_count > 0),
    }


def abstention_signals(
    *,
    answer: str,
    insufficient_evidence: bool,
    unsupported: bool,
) -> dict[str, Any]:
    """Lightweight lexical signals for unsupported / abstention review."""
    lower = (answer or "").lower()
    phrases = (
        "não contém informação suficiente",
        "nao contem informacao suficiente",
        "evidências recuperadas não estabelecem",
        "evidencias recuperadas nao estabelecem",
        "não estabelecem",
        "nao estabelecem",
        "não é possível",
        "nao e possivel",
        "não há evidência",
        "nao ha evidencia",
        "não dispõe",
        "nao dispoe",
    )
    matched = [phrase for phrase in phrases if phrase in lower]
    return {
        "unsupported_gold": unsupported,
        "insufficient_evidence_flag": insufficient_evidence,
        "abstention_phrase_matched": matched,
        "has_abstention_phrase": bool(matched),
    }
