import logging
from dataclasses import dataclass
from typing import Any

from rag.query.intent import detect_query_intent
from rag.query.query_rewriter import QueryRewriteException, QueryRewriter
from rag.retrieval.answerability import (
    adjust_definition_candidate_score,
    retrieval_definition_boost,
)
from rag.retrieval.bm25_index import Bm25DocumentIndex
from rag.vector.chroma_store import ChromaVectorStore, SearchResult

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
        self._bm25_index = Bm25DocumentIndex(self._documents)

    def retrieve(self, query: str, candidate_k: int) -> list[SearchResult]:
        if not query or not query.strip():
            raise HybridRetrieverException("Cannot retrieve with an empty query.")
        if candidate_k <= 0:
            raise HybridRetrieverException(
                "Hybrid retrieval parameter 'candidate_k' must be greater than zero."
            )

        rewritten_queries, intent, used_llm, rejected, llm_error, sources = (
            self._prepare_queries(query)
        )
        dense_weight, sparse_weight = self._get_weights_for_intent(intent)
        global_best: dict[str, HybridCandidate] = {}

        for retrieval_query in rewritten_queries:
            merged = self._merge_results(
                dense_results=self.vector_store.similarity_search_with_scores(
                    query=retrieval_query, k=candidate_k
                ),
                sparse_results=self._bm25_index.top_k(retrieval_query, candidate_k),
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                retrieval_intent=intent,
                retrieval_query=retrieval_query,
            )
            for candidate in self._apply_metadata_filter(
                self._apply_candidate_adjustments(merged, intent)
            ):
                key = self._result_key(candidate.metadata, candidate.content)
                existing = global_best.get(key)
                if existing is None or candidate.combined_score > existing.combined_score:
                    global_best[key] = candidate

        ranked = sorted(global_best.values(), key=lambda c: c.combined_score, reverse=True)
        return [
            self._to_search_result(
                candidate,
                query=query,
                intent=intent,
                rewritten_queries=rewritten_queries,
                used_llm=used_llm,
                rejected=rejected,
                llm_error=llm_error,
                sources=sources,
            )
            for candidate in ranked[:candidate_k]
        ]

    def _to_search_result(
        self,
        candidate: HybridCandidate,
        *,
        query: str,
        intent: str,
        rewritten_queries: list[str],
        used_llm: bool,
        rejected: list[str],
        llm_error: str | None,
        sources: dict[str, str],
    ) -> SearchResult:
        matched = candidate.metadata.get("matched_query", query)
        return SearchResult(
            content=candidate.content,
            metadata={
                **candidate.metadata,
                "dense_score": candidate.dense_score,
                "bm25_score": candidate.bm25_score,
                "answerability_boost": candidate.answerability_boost,
                "combined_score": candidate.combined_score,
                "retrieval_intent": intent,
                "rewritten_queries": rewritten_queries,
                "matched_query": matched,
                "original_query": query,
                "used_llm_query_rewrite": used_llm,
                "rejected_llm_queries": rejected,
                "llm_error": llm_error,
                "rewrite_source": sources.get(str(matched), "original"),
            },
            score=candidate.combined_score,
        )

    def _prepare_queries(
        self, query: str
    ) -> tuple[list[str], str, bool, list[str], str | None, dict[str, str]]:
        try:
            if self.query_rewriter is not None:
                result = self.query_rewriter.rewrite(query)
                return (
                    result.rewritten_queries or [query],
                    result.intent,
                    result.used_llm,
                    result.rejected_llm_queries,
                    result.llm_error,
                    result.rewrite_sources,
                )
        except QueryRewriteException as exc:
            logger.warning("Query rewriting failed, falling back to original query: %s", exc)
            fallback = query.strip()
            return [fallback], detect_query_intent(query), False, [], str(exc), {
                fallback: "original"
            }
        fallback = query.strip()
        return [fallback], detect_query_intent(query), False, [], None, {fallback: "original"}

    def _load_documents_from_chroma(self) -> list[dict[str, Any]]:
        try:
            raw = self.vector_store._client.get(include=["documents", "metadatas"])
            docs = raw.get("documents", []) or []
            metas = raw.get("metadatas", []) or []
            return [
                {"content": content, "metadata": metadata or {}}
                for content, metadata in zip(docs, metas)
                if content and not self._is_boilerplate_or_garbage(content, metadata or {})
            ]
        except Exception as exc:
            raise HybridRetrieverException("Failed to load documents from Chroma.") from exc

    def _get_weights_for_intent(self, intent: str) -> tuple[float, float]:
        if intent == "definition":
            return 0.4, 0.6
        return 0.6, 0.4

    def _is_boilerplate_or_garbage(self, text: str, metadata: dict[str, Any]) -> bool:
        if str(metadata.get("section_type", "body")).lower() == "references":
            return True
        if str(metadata.get("retrieval_quality", "normal")).lower() == "low":
            return True
        lower_text = text.lower()
        return any(
            indicator in lower_text
            for indicator in (
                "copyright",
                "permission to make digital or hard copies",
                "all rights reserved",
            )
        )

    def _normalize_dense(self, distance: float) -> float:
        return 1.0 / (1.0 + max(distance, 0.0))

    def _result_key(self, metadata: dict[str, Any], content: str) -> str:
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return (
            f"{metadata.get('source', 'unknown-source')}:"
            f"{metadata.get('page', 'unknown-page')}:"
            f"{metadata.get('chunk_index', 'unknown-chunk')}:"
            f"{hash(content)}"
        )

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
            key = self._result_key(result.metadata, result.content)
            metadata = dict(result.metadata)
            metadata["matched_query"] = retrieval_query
            merged[key] = {
                "content": result.content,
                "metadata": metadata,
                "dense_score": self._normalize_dense(result.score or 0.0),
                "bm25_score": 0.0,
            }
        for item in sparse_results:
            key = self._result_key(item["metadata"], item["content"])
            bm25_score = float(item["bm25_score"])
            if key not in merged:
                metadata = dict(item["metadata"])
                metadata["matched_query"] = retrieval_query
                merged[key] = {
                    "content": item["content"],
                    "metadata": metadata,
                    "dense_score": 0.0,
                    "bm25_score": bm25_score,
                }
            else:
                merged[key]["bm25_score"] = max(merged[key]["bm25_score"], bm25_score)

        content_penalties = {"mixed": 0.12, "table": 0.25, "boilerplate": 0.30}
        candidates: list[HybridCandidate] = []
        for value in merged.values():
            dense_score = float(value["dense_score"])
            bm25_score = float(value["bm25_score"])
            content = value["content"]
            metadata = value["metadata"]
            boost = (
                retrieval_definition_boost(content) if retrieval_intent == "definition" else 0.0
            )
            combined = dense_weight * dense_score + sparse_weight * bm25_score + boost
            combined -= content_penalties.get(
                str(metadata.get("content_type", "normal")).lower(), 0.0
            )
            candidates.append(
                HybridCandidate(
                    content=content,
                    metadata=metadata,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    answerability_boost=boost,
                    combined_score=max(combined, 0.0),
                )
            )
        return candidates

    def _apply_candidate_adjustments(
        self,
        candidates: list[HybridCandidate],
        retrieval_intent: str,
    ) -> list[HybridCandidate]:
        if retrieval_intent != "definition":
            return candidates
        return [
            HybridCandidate(
                content=c.content,
                metadata=c.metadata,
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                answerability_boost=c.answerability_boost,
                combined_score=adjust_definition_candidate_score(c.content, c.combined_score),
            )
            for c in candidates
        ]

    def _score_retrieval_answerability(self, text: str, intent: str) -> float:
        """Kept for focused unit tests of retrieval-time definition cues."""
        if intent != "definition":
            return 0.0
        return retrieval_definition_boost(text)

    def _apply_metadata_filter(
        self, candidates: list[HybridCandidate]
    ) -> list[HybridCandidate]:
        return [
            c
            for c in candidates
            if c.metadata.get("section_type", "body") != "references"
            and c.metadata.get("retrieval_quality", "normal") != "low"
            and str(c.metadata.get("content_type", "normal")).lower()
            not in {"table", "boilerplate"}
        ]
