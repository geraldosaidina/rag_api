import logging
import re
from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder

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
            intent = self._detect_query_intent(query)
            scored_items: list[RerankResult] = []
            for candidate, score in zip(candidates, scores):
                rerank_score = float(score)
                answerability_score = self.score_answerability(
                    query=query,
                    chunk=candidate.content,
                    intent=intent,
                    metadata=candidate.metadata,
                )
                final_score = rerank_score + answerability_score
                scored_items.append(
                    RerankResult(
                        content=candidate.content,
                        metadata=candidate.metadata,
                        original_score=candidate.score,
                        rerank_score=rerank_score,
                        answerability_score=answerability_score,
                        final_score=final_score,
                    )
                )

            ranked_items = sorted(
                scored_items,
                key=lambda item: item.final_score,
                reverse=True,
            )
            return ranked_items[:top_k]
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
        if not chunk or not chunk.strip():
            return 0.0
        if intent != "definition":
            return 0.0

        positive = self._definition_positive_score(chunk)
        penalty = self._definition_penalty_score(chunk, metadata or {})
        score = positive - penalty
        return max(min(score, 0.25), -0.30)

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

    def _definition_positive_score(self, chunk: str) -> float:
        lower = chunk.lower()
        score = 0.0

        phrase_weights = [
            ("is defined as", 0.12),
            ("can be defined as", 0.12),
            ("refers to", 0.10),
            ("is a method that", 0.09),
            ("is a framework that", 0.09),
            ("rag addresses", 0.10),
            ("rag models", 0.08),
            ("rag systems", 0.07),
            ("by integrating", 0.06),
            ("by combining", 0.06),
            ("introducing a retrieval component", 0.09),
        ]
        for phrase, value in phrase_weights:
            if phrase in lower:
                score += value

        # Explanation-like sentence bonus.
        sentence_candidates = re.split(r"(?<=[.!?])\s+", chunk)
        for sentence in sentence_candidates:
            word_count = len(re.findall(r"[A-Za-z0-9]+", sentence))
            if word_count < 14:
                continue
            sentence_lower = sentence.lower()
            if re.search(
                r"\b(is|are|means|uses|leverages|integrates|combines|augments|retrieves|generates)\b",
                sentence_lower,
            ):
                score += 0.05
                break

        # Subject + action co-occurrence bonus.
        has_subject = re.search(r"\b(rag|model|models|system|systems)\b", lower) is not None
        has_action = re.search(
            r"\b(augment|augments|integrate|integrates|retrieve|retrieves|generate|generates)\b",
            lower,
        ) is not None
        if has_subject and has_action:
            score += 0.05

        return min(score, 0.25)

    def _definition_penalty_score(self, chunk: str, metadata: dict[str, Any]) -> float:
        lower = chunk.lower().lstrip()
        penalty = 0.0

        weak_prefixes = [
            ("this paper", 0.12),
            ("in this survey", 0.12),
            ("we discuss", 0.10),
            ("open issues", 0.20),
            ("future directions", 0.18),
        ]
        for prefix, value in weak_prefixes:
            if lower.startswith(prefix):
                penalty += value
                break

        if str(metadata.get("content_type", "normal")).lower() == "mixed":
            penalty += 0.10

        return min(penalty, 0.30)
