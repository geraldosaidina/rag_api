"""In-memory BM25 index over Chroma documents used by hybrid retrieval."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "does", "do", "how", "what", "by", "with", "as", "from",
    "de", "da", "do", "das", "dos", "um", "uma", "o", "os", "as", "que", "em",
    "para", "por", "com", "como", "são", "sao", "é", "e", "sobre",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9à-ÿ]+", text.lower().replace("-", " "))
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def normalize_bm25(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    if maximum <= minimum:
        return [0.0 for _ in values]
    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


class Bm25DocumentIndex:
    def __init__(self, documents: list[dict[str, Any]]):
        self._documents = documents
        tokenized = [tokenize(item["content"]) for item in documents]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def top_k(self, query: str, k: int) -> list[dict[str, Any]]:
        if not self._bm25 or not self._documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        indexed = sorted(
            enumerate(self._bm25.get_scores(query_tokens).tolist()),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        normalized = normalize_bm25([float(score) for _, score in indexed])
        results: list[dict[str, Any]] = []
        for (doc_idx, _), score in zip(indexed, normalized):
            doc = self._documents[doc_idx]
            results.append(
                {"content": doc["content"], "metadata": doc["metadata"], "bm25_score": score}
            )
        return results
