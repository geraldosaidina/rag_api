"""Helpers for the Phase 2 generation benchmark runner."""

from __future__ import annotations

from typing import Any

from evaluation.metrics.retrieval import mean_of


def gold_pages(gold_payload: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    pages: dict[str, dict[str, list[int]]] = {}
    for item in gold_payload["questions"]:
        by_pfc: dict[str, list[int]] = {}
        for evidence in item.get("evidence") or []:
            pfc_id = str(evidence["pfc_id"])
            by_pfc.setdefault(pfc_id, [])
            for page in evidence.get("pdf_pages") or []:
                by_pfc[pfc_id].append(int(page))
        pages[str(item["id"])] = by_pfc
    return pages


def answer_notes(gold_payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["id"]): str(item["expected"].get("answer_notes") or "")
        for item in gold_payload["questions"]
    }


def serialize_sources(result) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": source.citation_id,
            "source": source.source,
            "page": source.page,
            "chunk_index": source.chunk_index,
            "chunk_id": source.chunk_id,
            "excerpt": source.excerpt,
            "pfc_id": source.pfc_id,
            "title": source.title,
            "authors": list(source.authors),
            "year": source.year,
        }
        for source in result.sources
    ]


def serialize_evidence(result) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": item.citation_id,
            "content": item.content,
            "source": item.source,
            "page": item.page,
            "chunk_index": item.chunk_index,
            "chunk_id": item.chunk_id,
            "excerpt": item.excerpt,
            "pfc_id": item.pfc_id,
        }
        for item in result.evidence_items
    ]


def aggregate_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if not row["unsupported"]]
    unsupported = [row for row in rows if row["unsupported"]]
    return {
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "supported_macro": {
            "citation_doc_precision": mean_of(
                row["citation_metrics"]["citation_doc_precision"] for row in supported
            ),
            "citation_doc_recall": mean_of(
                row["citation_metrics"]["citation_doc_recall"] for row in supported
            ),
            "citation_page_precision": mean_of(
                row["citation_metrics"]["citation_page_precision"] for row in supported
            ),
            "mean_total_duration_ms": mean_of(row["total_duration_ms"] for row in supported),
            "mean_generation_duration_ms": mean_of(
                row["generation_duration_ms"] for row in supported
            ),
            "llm_faithfulness": mean_of(
                (row.get("llm_judge") or {}).get("faithfulness") for row in supported
            ),
            "llm_answer_relevance": mean_of(
                (row.get("llm_judge") or {}).get("answer_relevance") for row in supported
            ),
            "llm_citation_semantic_support": mean_of(
                (row.get("llm_judge") or {}).get("citation_semantic_support")
                for row in supported
            ),
            "human_review_rate": mean_of(
                1.0 if (row.get("llm_judge") or {}).get("needs_human_review") else 0.0
                for row in supported
                if row.get("llm_judge") is not None
            ),
        },
        "unsupported_macro": {
            "insufficient_evidence_rate": mean_of(
                1.0 if row["insufficient_evidence"] else 0.0 for row in unsupported
            ),
            "abstention_phrase_rate": mean_of(
                1.0 if row["abstention"]["has_abstention_phrase"] else 0.0
                for row in unsupported
            ),
            "llm_unsupported_handling": mean_of(
                (row.get("llm_judge") or {}).get("unsupported_handling")
                for row in unsupported
            ),
            "human_review_rate": mean_of(
                1.0 if (row.get("llm_judge") or {}).get("needs_human_review") else 0.0
                for row in unsupported
                if row.get("llm_judge") is not None
            ),
        },
        "unsupported_question_ids": [row["question_id"] for row in unsupported],
    }
