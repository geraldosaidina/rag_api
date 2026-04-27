import json
import logging
import re
from dataclasses import dataclass

from rag.llm.ollama_client import LLMException, OllamaLLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryRewriteConfig:
    enabled: bool = True
    use_llm: bool = True
    max_rewrites: int = 5
    include_original: bool = True


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    intent: str
    rewritten_queries: list[str]
    used_llm: bool


class QueryRewriteException(Exception):
    """Raised when query rewriting cannot be completed."""


class QueryRewriter:
    def __init__(
        self,
        llm_client: OllamaLLMClient | None = None,
        config: QueryRewriteConfig | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or QueryRewriteConfig()

    def rewrite(self, query: str) -> QueryRewriteResult:
        if not query or not query.strip():
            raise QueryRewriteException("Cannot rewrite an empty query.")

        original_query = query.strip()
        intent = self._detect_intent(original_query)

        rule_rewrites = self._rule_based_rewrites(original_query, intent)
        rewrites: list[str] = list(rule_rewrites)
        used_llm = False

        if self.config.enabled and self.config.use_llm and self.llm_client is not None:
            llm_rewrites = self._llm_rewrites(original_query, intent)
            if llm_rewrites:
                rewrites.extend(llm_rewrites)
                used_llm = True

        deduped = self._dedupe_and_clean(rewrites)
        if self.config.include_original:
            deduped = self._dedupe_and_clean([original_query, *deduped])

        max_rewrites = max(self.config.max_rewrites, 0)
        if self.config.include_original:
            final_rewrites = deduped[: max_rewrites + 1]
        else:
            final_rewrites = deduped[:max_rewrites]

        if not final_rewrites:
            final_rewrites = [original_query]

        return QueryRewriteResult(
            original_query=original_query,
            intent=intent,
            rewritten_queries=final_rewrites,
            used_llm=used_llm,
        )

    def _detect_intent(self, query: str) -> str:
        lower = query.lower()
        components_terms = ["components", "parts", "elements", "architecture", "modules"]
        definition_terms = [
            "define",
            "definition",
            "what is",
            "what are",
            "meaning of",
            "refers to",
        ]
        comparison_terms = ["compare", "difference", "differentiate", "versus", "vs"]
        mechanism_terms = ["how does", "how do", "how can", "reduce", "improve", "affect"]

        if any(term in lower for term in components_terms):
            return "components"
        if any(term in lower for term in definition_terms):
            return "definition"
        if any(term in lower for term in comparison_terms):
            return "comparison"
        if any(term in lower for term in mechanism_terms):
            return "mechanism"
        return "general"

    def _rule_based_rewrites(self, query: str, intent: str) -> list[str]:
        rewrites = [query]
        lower = query.lower()
        rag_related = (
            "retrieval-augmented generation" in lower
            or "retrieval augmented generation" in lower
            or re.search(r"\brag\b", lower) is not None
        )

        if not rag_related:
            return self._dedupe_and_clean(rewrites)

        rewrites.extend(
            [
                "Retrieval-Augmented Generation",
                "RAG",
            ]
        )

        if intent == "definition":
            rewrites.extend(
                [
                    "Retrieval-Augmented Generation is defined as",
                    "RAG refers to",
                    "RAG combines retrieval and generation",
                    "RAG uses external knowledge during generation",
                    "RAG introduces a retrieval component",
                ]
            )
        elif intent == "components":
            rewrites.extend(
                [
                    "core components of Retrieval-Augmented Generation",
                    "RAG components retrieval knowledge integration answer generation",
                    "RAG framework consists of",
                    "knowledge sourcing embedding retrieval integration generation citation",
                ]
            )
        elif intent == "mechanism":
            rewrites.extend(
                [
                    "RAG reduces hallucinations by grounding generation in retrieved context",
                    "RAG improves factuality using external knowledge",
                    "retrieved context reduces hallucinations",
                    "RAG retrieves relevant information before generation",
                ]
            )
        elif intent == "comparison":
            rewrites.extend(
                [
                    "compare Retrieval-Augmented Generation with standard language models",
                    "difference between RAG and parametric-only generation",
                    "RAG versus non-retrieval generation approach",
                ]
            )
        else:
            rewrites.extend(
                [
                    "Retrieval-Augmented Generation RAG overview",
                ]
            )

        return self._dedupe_and_clean(rewrites)

    def _llm_rewrites(self, query: str, intent: str) -> list[str]:
        if self.llm_client is None:
            return []

        system_prompt = (
            "You rewrite user questions into search queries for a RAG retrieval system. "
            "You do not answer questions. You do not add facts. You preserve technical "
            "terms. Return strict JSON only."
        )
        user_prompt = (
            f"Original question: {query}\n"
            f"Intent: {intent}\n\n"
            f"Create up to {self.config.max_rewrites} retrieval-optimized search queries "
            "that are likely to match wording inside academic documents.\n\n"
            'Return JSON only in this form: {"queries": ["q1", "q2"]}'
        )

        try:
            response = self.llm_client.invoke_messages(
                [
                    ("system", system_prompt),
                    ("user", user_prompt),
                ]
            )
            return self._parse_llm_json_queries(response.content)
        except (LLMException, QueryRewriteException) as exc:
            logger.warning("LLM rewrite failed, falling back to rule-based rewrites: %s", exc)
            return []
        except Exception as exc:
            logger.warning("Unexpected LLM rewrite error, using fallback: %s", exc)
            return []

    def _parse_llm_json_queries(self, raw_content: str) -> list[str]:
        try:
            parsed = json.loads(raw_content.strip())
        except json.JSONDecodeError as exc:
            raise QueryRewriteException("LLM rewrite output is not valid JSON.") from exc

        queries = parsed.get("queries") if isinstance(parsed, dict) else None
        if not isinstance(queries, list):
            raise QueryRewriteException("LLM rewrite JSON must include a 'queries' list.")

        cleaned: list[str] = []
        for item in queries:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    cleaned.append(stripped)

        deduped = self._dedupe_and_clean(cleaned)
        return deduped[: self.config.max_rewrites]

    def _dedupe_and_clean(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            stripped = value.strip()
            if not stripped:
                continue
            key = re.sub(r"\s+", " ", stripped).lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(stripped)
        return result
