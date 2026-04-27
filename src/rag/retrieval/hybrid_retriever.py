import logging
import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from rag.query.query_rewriter import QueryRewriteException, QueryRewriter
from rag.vector.chroma_store import ChromaException, ChromaVectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridCandidate:
    content: str
    metadata: dict[str, Any]
    dense_score: float
    bm25_score: float
    answerability_boost: float
    combined_score: float


class HybridRetrieverException(Exception):
    """Raised when hybrid retrieval cannot complete successfully."""


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_function: Any,
        query_rewriter: QueryRewriter | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.query_rewriter = query_rewriter
        self._documents = self._load_documents_from_chroma()
        self._tokenized_corpus = [
            self._tokenize(item["content"]) for item in self._documents
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def retrieve(self, query: str, candidate_k: int) -> list[SearchResult]:
        if not query or not query.strip():
            raise HybridRetrieverException("Cannot retrieve with an empty query.")
        if candidate_k <= 0:
            raise HybridRetrieverException(
                "Hybrid retrieval parameter 'candidate_k' must be greater than zero."
            )

        rewritten_queries: list[str]
        used_llm_query_rewrite = False
        try:
            if self.query_rewriter is not None:
                rewrite_result = self.query_rewriter.rewrite(query)
                rewritten_queries = rewrite_result.rewritten_queries or [query]
                retrieval_intent = rewrite_result.intent
                used_llm_query_rewrite = rewrite_result.used_llm
            else:
                rewritten_queries = [self._expand_query(query)]
                retrieval_intent = self._detect_query_intent(query)
        except QueryRewriteException as exc:
            logger.warning("Query rewriting failed, falling back to expanded query: %s", exc)
            rewritten_queries = [self._expand_query(query)]
            retrieval_intent = self._detect_query_intent(query)

        dense_weight, sparse_weight = self._get_weights_for_intent(retrieval_intent)
        global_best_candidates: dict[str, HybridCandidate] = {}

        for retrieval_query in rewritten_queries:
            dense_results = self.vector_store.similarity_search_with_scores(
                query=retrieval_query,
                k=candidate_k,
            )
            sparse_results = self._bm25_top_k(query=retrieval_query, k=candidate_k)

            merged = self._merge_results(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                retrieval_intent=retrieval_intent,
                retrieval_query=retrieval_query,
            )
            adjusted = self._apply_candidate_adjustments(
                candidates=merged,
                retrieval_intent=retrieval_intent,
            )
            filtered = self._apply_metadata_filter(adjusted)

            for candidate in filtered:
                key = self._result_key(candidate.metadata, candidate.content)
                existing = global_best_candidates.get(key)
                if existing is None or candidate.combined_score > existing.combined_score:
                    global_best_candidates[key] = candidate

        sorted_candidates = sorted(
            global_best_candidates.values(),
            key=lambda candidate: candidate.combined_score,
            reverse=True,
        )[:candidate_k]

        return [
            SearchResult(
                content=candidate.content,
                metadata={
                    **candidate.metadata,
                    "dense_score": candidate.dense_score,
                    "bm25_score": candidate.bm25_score,
                    "answerability_boost": candidate.answerability_boost,
                    "combined_score": candidate.combined_score,
                    "retrieval_intent": retrieval_intent,
                    "rewritten_queries": rewritten_queries,
                    "matched_query": candidate.metadata.get("matched_query", query),
                    "original_query": query,
                    "used_llm_query_rewrite": used_llm_query_rewrite,
                },
                score=candidate.combined_score,
            )
            for candidate in sorted_candidates
        ]

    def _load_documents_from_chroma(self) -> list[dict[str, Any]]:
        try:
            raw = self.vector_store._client.get(include=["documents", "metadatas"])
            documents = raw.get("documents", []) or []
            metadatas = raw.get("metadatas", []) or []
            result: list[dict[str, Any]] = []
            for content, metadata in zip(documents, metadatas):
                if not content:
                    continue
                metadata = metadata or {}
                if self._is_boilerplate_or_garbage(content, metadata):
                    continue
                result.append({"content": content, "metadata": metadata})
            return result
        except Exception as exc:
            raise HybridRetrieverException("Failed to load documents from Chroma.") from exc

    def _tokenize(self, text: str) -> list[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "to",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "does",
            "do",
            "how",
            "what",
            "by",
            "with",
            "as",
            "from",
        }
        normalized = text.lower().replace("-", " ")
        tokens = re.findall(r"[a-z0-9]+", normalized)
        return [token for token in tokens if token not in stopwords]

    def _expand_query(self, query: str) -> str:
        expanded_parts = [query.strip()]
        lower = query.lower()
        has_rag_phrase = (
            "retrieval-augmented generation" in lower
            or "retrieval augmented generation" in lower
        )
        has_rag_term = bool(re.search(r"\brag\b", lower))

        if has_rag_phrase or has_rag_term:
            expanded_parts.append("retrieval augmented generation rag definition explanation")

        return " ".join(expanded_parts).strip()

    def _detect_query_intent(self, query: str) -> str:
        lower = query.lower()
        definition_terms = [
            "define",
            "definition",
            "what is",
            "what are",
            "meaning of",
            "refers to",
        ]
        comparison_terms = [
            "compare",
            "difference",
            "differentiate",
            "versus",
            "vs",
        ]
        if any(term in lower for term in definition_terms):
            return "definition"
        if any(term in lower for term in comparison_terms):
            return "comparison"
        return "general"

    def _get_weights_for_intent(self, intent: str) -> tuple[float, float]:
        if intent == "definition":
            return 0.4, 0.6
        if intent == "comparison":
            return 0.6, 0.4
        return 0.6, 0.4

    def _is_boilerplate_or_garbage(self, text: str, metadata: dict[str, Any]) -> bool:
        section_type = str(metadata.get("section_type", "body")).lower()
        retrieval_quality = str(metadata.get("retrieval_quality", "normal")).lower()
        if section_type == "references":
            return True
        if retrieval_quality == "low":
            return True

        lower_text = text.lower()
        garbage_indicators = [
            "copyright",
            "permission to make digital or hard copies",
            "conference acronym",
            "woodstock, ny",
            "priprint",
            "all rights reserved",
        ]
        return any(indicator in lower_text for indicator in garbage_indicators)

    def _normalize_dense(self, distance: float) -> float:
        return 1.0 / (1.0 + max(distance, 0.0))

    def _normalize_bm25(self, values: list[float]) -> list[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            return [0.0 for _ in values]
        scale = maximum - minimum
        return [(value - minimum) / scale for value in values]

    def _result_key(self, metadata: dict[str, Any], content: str) -> str:
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        source = metadata.get("source", "unknown-source")
        page = metadata.get("page", "unknown-page")
        chunk_index = metadata.get("chunk_index", "unknown-chunk")
        return f"{source}:{page}:{chunk_index}:{hash(content)}"

    def _bm25_top_k(self, query: str, k: int) -> list[dict[str, Any]]:
        if not self._bm25 or not self._documents:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens).tolist()
        indexed_scores = list(enumerate(raw_scores))
        indexed_scores.sort(key=lambda item: item[1], reverse=True)
        top_items = indexed_scores[:k]
        bm25_values = [float(score) for _, score in top_items]
        normalized = self._normalize_bm25(bm25_values)

        results: list[dict[str, Any]] = []
        for (doc_idx, _), normalized_score in zip(top_items, normalized):
            doc = self._documents[doc_idx]
            results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "bm25_score": normalized_score,
                }
            )
        return results

    def _merge_results(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[dict[str, Any]],
        dense_weight: float,
        sparse_weight: float,
        retrieval_intent: str,
        retrieval_query: str,
    ) -> list[HybridCandidate]:
        merged: dict[str, dict[str, Any]] = {}

        for result in dense_results:
            dense_score = self._normalize_dense(result.score or 0.0)
            key = self._result_key(result.metadata, result.content)
            metadata = dict(result.metadata)
            metadata["matched_query"] = retrieval_query
            merged[key] = {
                "content": result.content,
                "metadata": metadata,
                "dense_score": dense_score,
                "bm25_score": 0.0,
            }

        for item in sparse_results:
            content = item["content"]
            metadata = item["metadata"]
            bm25_score = float(item["bm25_score"])
            key = self._result_key(metadata, content)
            if key not in merged:
                item_metadata = dict(metadata)
                item_metadata["matched_query"] = retrieval_query
                merged[key] = {
                    "content": content,
                    "metadata": item_metadata,
                    "dense_score": 0.0,
                    "bm25_score": bm25_score,
                }
            else:
                merged[key]["bm25_score"] = max(merged[key]["bm25_score"], bm25_score)

        candidates: list[HybridCandidate] = []
        for value in merged.values():
            dense_score = float(value["dense_score"])
            bm25_score = float(value["bm25_score"])
            content = value["content"]
            metadata = value["metadata"]
            combined = dense_weight * dense_score + sparse_weight * bm25_score
            answerability_boost = self._score_retrieval_answerability(
                text=content,
                intent=retrieval_intent,
            )
            combined += answerability_boost

            content_type = str(metadata.get("content_type", "normal")).lower()
            if content_type == "mixed":
                combined -= 0.12
            elif content_type == "table":
                combined -= 0.25
            elif content_type == "boilerplate":
                combined -= 0.30

            combined = max(combined, 0.0)
            candidates.append(
                HybridCandidate(
                    content=content,
                    metadata=metadata,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    answerability_boost=answerability_boost,
                    combined_score=combined,
                )
            )
        return candidates

    def _apply_candidate_adjustments(
        self,
        candidates: list[HybridCandidate],
        retrieval_intent: str,
    ) -> list[HybridCandidate]:
        adjusted: list[HybridCandidate] = []
        for candidate in candidates:
            score = candidate.combined_score
            lower_content = candidate.content.lower().strip()

            if retrieval_intent == "definition":
                weak_prefixes = [
                    "and opportunities",
                    "this paper concludes",
                    "open issues",
                ]
                if any(lower_content.startswith(prefix) for prefix in weak_prefixes):
                    score -= 0.08

                definition_boost_terms = [
                    ("rag addresses", 0.12),
                    ("rag models augment", 0.12),
                    ("retrieval component", 0.08),
                    ("external knowledge", 0.08),
                    ("is defined as", 0.10),
                    ("refers to", 0.08),
                    ("can be defined as", 0.10),
                    ("consists of", 0.05),
                ]
                boost = 0.0
                for phrase, value in definition_boost_terms:
                    if phrase in lower_content:
                        boost += value
                score += min(boost, 0.12)

            adjusted.append(
                HybridCandidate(
                    content=candidate.content,
                    metadata=candidate.metadata,
                    dense_score=candidate.dense_score,
                    bm25_score=candidate.bm25_score,
                    answerability_boost=candidate.answerability_boost,
                    combined_score=max(score, 0.0),
                )
            )
        return adjusted

    def _score_retrieval_answerability(self, text: str, intent: str) -> float:
        if intent != "definition":
            return 0.0
        lower_text = text.lower()
        phrase_weights = [
            ("is defined as", 0.04),
            ("can be defined as", 0.04),
            ("refers to", 0.03),
            ("is a method", 0.03),
            ("is a framework", 0.03),
            ("rag addresses", 0.03),
            ("rag models", 0.02),
            ("rag systems", 0.02),
            ("retrieval component", 0.02),
            ("external knowledge", 0.02),
            ("by integrating", 0.02),
            ("by combining", 0.02),
            ("consists of", 0.02),
        ]
        score = 0.0
        for phrase, value in phrase_weights:
            if phrase in lower_text:
                score += value
        return min(max(score, 0.0), 0.15)

    def _apply_metadata_filter(
        self,
        candidates: list[HybridCandidate],
    ) -> list[HybridCandidate]:
        filtered: list[HybridCandidate] = []
        for candidate in candidates:
            section_type = candidate.metadata.get("section_type", "body")
            retrieval_quality = candidate.metadata.get("retrieval_quality", "normal")
            content_type = str(candidate.metadata.get("content_type", "normal")).lower()
            if section_type == "references":
                continue
            if retrieval_quality == "low":
                continue
            if content_type in {"table", "boilerplate"}:
                continue
            filtered.append(candidate)
        return filtered
