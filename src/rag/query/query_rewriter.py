import json
import logging
import re
from dataclasses import dataclass, field

from rag.llm.ollama_client import LLMException, OllamaLLMClient
from rag.query.intent import detect_query_intent
from rag.query.rewrite_helpers import acronym_expansions, extract_subject, intent_rule_rewrites

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_CONTEXT = (
    "The searchable collection contains academic final-course projects (PFCs) from "
    "Mozambican higher-education institutions. Projects span arbitrary computing and "
    "technology domains such as software engineering, artificial intelligence, finance, "
    "cybersecurity, networking, IoT, data science, and information systems. "
    "Use this corpus context only to disambiguate ambiguous terms; never invent a more "
    "specific topic than the user supplied."
)

DEFAULT_DOMAIN_EXCLUSION_TERMS = (
    "medical",
    "psychiatric",
    "psychiatry",
    "clinical",
    "schizophrenia",
    "neurological",
    "patient",
    "diagnosis",
    "médico",
    "medico",
    "psiquiátrico",
    "psiquiatrico",
)


@dataclass(frozen=True)
class QueryRewriteConfig:
    enabled: bool = True
    use_llm: bool = True
    max_rewrites: int = 3
    include_original: bool = True
    domain_context: str = DEFAULT_DOMAIN_CONTEXT
    reject_off_domain_rewrites: bool = True
    domain_exclusion_terms: tuple[str, ...] = DEFAULT_DOMAIN_EXCLUSION_TERMS


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    intent: str
    rewritten_queries: list[str]
    used_llm: bool
    rejected_llm_queries: list[str] = field(default_factory=list)
    llm_error: str | None = None
    rewrite_sources: dict[str, str] = field(default_factory=dict)


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
        intent = detect_query_intent(original_query)
        rule_rewrites = self._rule_based_rewrites(original_query, intent)

        llm_rewrites: list[str] = []
        rejected_llm_queries: list[str] = []
        llm_error: str | None = None
        used_llm = False

        if self.config.enabled and self.config.use_llm and self.llm_client is not None:
            llm_rewrites, rejected_llm_queries, llm_error = self._llm_rewrites(
                original_query,
                intent,
            )
            used_llm = bool(llm_rewrites)

        selected = self._select_balanced_rewrites(
            original_query=original_query,
            rule_rewrites=rule_rewrites,
            llm_rewrites=llm_rewrites,
        )
        return QueryRewriteResult(
            original_query=original_query,
            intent=intent,
            rewritten_queries=selected,
            used_llm=used_llm,
            rejected_llm_queries=rejected_llm_queries,
            llm_error=llm_error,
            rewrite_sources=self._assign_rewrite_sources(
                original_query, selected, rule_rewrites, llm_rewrites
            ),
        )

    def _detect_intent(self, query: str) -> str:
        return detect_query_intent(query)

    def _rule_based_rewrites(self, query: str, intent: str) -> list[str]:
        rewrites = list(acronym_expansions(query))
        rewrites.extend(intent_rule_rewrites(extract_subject(query), intent))
        return self._dedupe_and_clean(rewrites)

    def _llm_rewrites(
        self,
        query: str,
        intent: str,
    ) -> tuple[list[str], list[str], str | None]:
        if self.llm_client is None:
            return [], [], None

        system_prompt = (
            "You rewrite user questions into retrieval queries for a semantic search "
            "system over academic final-course projects (PFCs). "
            "Do not answer the question. "
            "Do not invent a more specific topic than the user supplied. "
            "Preserve entities, technical terms, and acronyms. "
            "Expand obvious acronyms only when justified by the query. "
            "Prefer conservative, document-style phrasing. "
            "Return strict JSON only."
        )
        user_prompt = (
            f"Original query: {query}\n"
            f"Intent: {intent}\n\n"
            f"Corpus context:\n{self.config.domain_context}\n\n"
            f"Generate up to {self.config.max_rewrites} retrieval-oriented query variants.\n\n"
            "Requirements:\n"
            "- preserve the original subject and terminology\n"
            "- keep Portuguese queries primarily in Portuguese and English in English\n"
            "- do not drift into unrelated domains\n"
            "- do not answer the question\n\n"
            'Return JSON only in this form: {"queries": ["q1", "q2"]}'
        )

        try:
            response = self.llm_client.invoke_messages(
                [("system", system_prompt), ("user", user_prompt)]
            )
            parsed = self._parse_llm_json_queries(response.content)
        except (LLMException, QueryRewriteException) as exc:
            logger.warning("LLM rewrite failed, falling back to rule-based rewrites: %s", exc)
            return [], [], str(exc)
        except Exception as exc:
            logger.warning("Unexpected LLM rewrite error, using fallback: %s", exc)
            return [], [], f"{type(exc).__name__}: {exc}"

        accepted: list[str] = []
        rejected: list[str] = []
        for candidate in parsed:
            if self._validate_llm_rewrite(query, candidate):
                accepted.append(candidate)
            else:
                rejected.append(candidate)
        return self._dedupe_and_clean(accepted), self._dedupe_and_clean(rejected), None

    def _validate_llm_rewrite(self, original_query: str, rewritten_query: str) -> bool:
        cleaned = rewritten_query.strip()
        if not cleaned or len(self._meaningful_tokens(cleaned)) < 2:
            return False
        if self._normalize_key(cleaned) == self._normalize_key(original_query):
            return False

        lower = cleaned.lower()
        original_lower = original_query.lower()
        if self.config.reject_off_domain_rewrites:
            for term in self.config.domain_exclusion_terms:
                if re.search(rf"\b{re.escape(term)}\b", lower) and term not in original_lower:
                    return False

        original_tokens = set(self._meaningful_tokens(original_query))
        rewrite_tokens = set(self._meaningful_tokens(cleaned))
        return bool(original_tokens & rewrite_tokens)

    def _select_balanced_rewrites(
        self,
        original_query: str,
        rule_rewrites: list[str],
        llm_rewrites: list[str],
    ) -> list[str]:
        max_rewrites = max(self.config.max_rewrites, 0)
        selected: list[str] = []
        if self.config.include_original:
            selected.append(original_query)
        selected.extend(rule_rewrites[:2])
        selected.extend(llm_rewrites)
        deduped = self._dedupe_and_clean(selected)
        if self.config.include_original:
            return deduped[: max_rewrites + 1]
        return deduped[:max_rewrites] or [original_query]

    def _assign_rewrite_sources(
        self,
        original_query: str,
        selected: list[str],
        rule_rewrites: list[str],
        llm_rewrites: list[str],
    ) -> dict[str, str]:
        rule_keys = {self._normalize_key(item) for item in rule_rewrites}
        llm_keys = {self._normalize_key(item) for item in llm_rewrites}
        original_key = self._normalize_key(original_query)
        sources: dict[str, str] = {}
        for item in selected:
            key = self._normalize_key(item)
            if key == original_key:
                sources[item] = "original"
            elif key in llm_keys:
                sources[item] = "llm"
            else:
                sources[item] = "rule"
        return sources

    def _parse_llm_json_queries(self, raw_content: str) -> list[str]:
        content = raw_content.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        if fenced:
            content = fenced.group(1)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise QueryRewriteException("LLM rewrite output is not valid JSON.") from exc

        queries = parsed.get("queries") if isinstance(parsed, dict) else None
        if not isinstance(queries, list):
            raise QueryRewriteException("LLM rewrite JSON must include a 'queries' list.")

        cleaned = [item.strip() for item in queries if isinstance(item, str) and item.strip()]
        return self._dedupe_and_clean(cleaned)[: self.config.max_rewrites]

    def _meaningful_tokens(self, text: str) -> list[str]:
        stopwords = {
            "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
            "was", "were", "does", "do", "how", "what", "by", "with", "as", "from",
            "de", "da", "do", "das", "dos", "um", "uma", "uns", "umas", "o", "os", "as",
            "que", "em", "no", "na", "nos", "nas", "para", "por", "com", "como",
            "são", "sao", "é", "e", "sobre", "quais", "qual",
        }
        tokens = re.findall(r"[a-z0-9à-ÿ]+", text.lower().replace("-", " "))
        return [token for token in tokens if token not in stopwords and len(token) > 1]

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip()).lower()

    def _dedupe_and_clean(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            stripped = value.strip()
            if not stripped:
                continue
            key = self._normalize_key(stripped)
            if key in seen:
                continue
            seen.add(key)
            result.append(stripped)
        return result

