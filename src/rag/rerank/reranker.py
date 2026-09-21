import logging
import re
from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder

from rag.query.intent import detect_query_intent
from rag.retrieval.answerability import rerank_definition_answerability
from rag.vector.chroma_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int | None = None
    device: str | None = None


@dataclass(frozen=True)
class RerankResult:
    content: str
    metadata: dict[str, Any]
    original_score: float | None
    rerank_score: float
    answerability_score: float
    final_score: float


class RerankerException(Exception):
    """Raised when reranking cannot be completed."""


class CrossEncoderReranker:
    """App-facing reranker adapter built on sentence-transformers CrossEncoder."""

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        try:
            logger.info(
                "Loading cross-encoder reranker model '%s'.",
                self.config.model_name,
            )
            self._model = CrossEncoder(
                model_name_or_path=self.config.model_name,
                max_length=self.config.max_length,
                device=self.config.device,
            )
        except Exception as exc:
            raise RerankerException(
                f"Failed to initialize reranker model '{self.config.model_name}'."
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[RerankResult]:
        if not query or not query.strip():
            raise RerankerException("Cannot rerank with an empty query.")
        if top_k <= 0:
            raise RerankerException("Rerank parameter 'top_k' must be greater than zero.")
        if not candidates:
            logger.info("No candidates provided for reranking.")
            return []

        logger.info(
            "Reranking %s candidates with model '%s' (top_k=%s).",
            len(candidates),
            self.config.model_name,
            top_k,
        )

        try:
            pairs = [(query, candidate.content) for candidate in candidates]
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RerankerException("Failed to compute reranker scores.") from exc

        try:
            intent = detect_query_intent(query)
            scored_items: list[RerankResult] = []
            for candidate, score in zip(candidates, scores):
                rerank_score = float(score)
                answerability_score = self.score_answerability(
                    query=query,
                    chunk=candidate.content,
                    intent=intent,
                    metadata=candidate.metadata,
                )
                scored_items.append(
                    RerankResult(
                        content=candidate.content,
                        metadata=candidate.metadata,
                        original_score=candidate.score,
                        rerank_score=rerank_score,
                        answerability_score=answerability_score,
                        final_score=rerank_score + answerability_score,
                    )
                )

            return sorted(
                scored_items,
                key=lambda item: item.final_score,
                reverse=True,
            )[:top_k]
        except Exception as exc:
            raise RerankerException("Failed to build reranked results.") from exc

    def score_answerability(
        self,
        query: str,
        chunk: str,
        intent: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        _ = query
        if intent != "definition":
            return 0.0
        return rerank_definition_answerability(chunk, metadata)
