"""Document-level retrieval metrics for the Phase 1 benchmark.

Relevance is taken from gold ``primary_pfc_ids``. Chunk rankings are collapsed
to the first occurrence of each ``pfc_id``.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def collapse_to_document_ranking(
    results: Sequence[Any],
) -> list[str]:
    """Return ordered unique ``pfc_id`` values from chunk-level hits."""
    ranked: list[str] = []
    seen: set[str] = set()
    for item in results:
        metadata = getattr(item, "metadata", None)
        if metadata is None and isinstance(item, dict):
            metadata = item.get("metadata") or item
        if not isinstance(metadata, dict):
            continue
        raw = metadata.get("pfc_id")
        if raw is None or raw == "":
            continue
        pfc_id = str(raw)
        if pfc_id in seen:
            continue
        seen.add(pfc_id)
        ranked.append(pfc_id)
    return ranked


def precision_at_k(ranked_docs: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant_set = {str(item) for item in relevant}
    top = list(ranked_docs)[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant_set)
    return hits / float(k)


def recall_at_k(ranked_docs: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant_set = {str(item) for item in relevant}
    if not relevant_set:
        return None
    top = list(ranked_docs)[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_set)
    return hits / float(len(relevant_set))


def mean_reciprocal_rank(
    ranked_docs: Sequence[str], relevant: Iterable[str]
) -> float | None:
    relevant_set = {str(item) for item in relevant}
    if not relevant_set:
        return None
    for index, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_set:
            return 1.0 / float(index)
    return 0.0


def score_ranking(
    ranked_docs: Sequence[str],
    relevant: Sequence[str],
    *,
    unsupported: bool,
) -> dict[str, float | None]:
    """Compute P@5, R@5, R@10, MRR. Unsupported queries get null IR scores."""
    if unsupported:
        return {
            "precision_at_5": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
        }
    return {
        "precision_at_5": precision_at_k(ranked_docs, relevant, 5),
        "recall_at_5": recall_at_k(ranked_docs, relevant, 5),
        "recall_at_10": recall_at_k(ranked_docs, relevant, 10),
        "mrr": mean_reciprocal_rank(ranked_docs, relevant),
    }


def mean_of(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / float(len(numeric))
