"""Topic-neutral regression checks for Slice 1 generalization.

Run:
  uv run python src/scripts/test_slice1_baseline.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.query.intent import detect_query_intent
from rag.query.query_rewriter import QueryRewriteConfig, QueryRewriter
from rag.retrieval.hybrid_retriever import HybridRetriever


BANNED_PRODUCTION_PHRASES = (
    "rag addresses",
    "rag combines",
    "rag models augment",
    "introducing a retrieval component",
    "hallucinations in rag systems",
    "Retrieval-Augmented Generation is defined as",
    "rag refers to",
    "rag versus",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_production_sources_are_topic_neutral() -> None:
    paths = [
        PROJECT_SRC / "rag" / "query" / "query_rewriter.py",
        PROJECT_SRC / "rag" / "retrieval" / "hybrid_retriever.py",
        PROJECT_SRC / "rag" / "rerank" / "reranker.py",
        PROJECT_SRC / "rag" / "ingest" / "chunker.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    lower = combined.lower()
    for phrase in BANNED_PRODUCTION_PHRASES:
        _assert(phrase.lower() not in lower, f"Banned phrase still present: {phrase}")

    _assert(
        "A Survey on Knowledge-Oriented Retrieval-Augmented Generation" not in combined,
        "Survey-title cleaning must not remain in production chunker.",
    )


def test_intent_matrix() -> None:
    cases = {
        "Define machine learning.": "definition",
        "Defina computação em nuvem.": "definition",
        "What are the components of a recommendation system?": "components",
        "Quais são os componentes de uma arquitectura cliente-servidor?": "components",
        "How does biometric authentication work?": "mechanism",
        "Como funciona a autenticação biométrica?": "mechanism",
        "Compare SQL and NoSQL databases.": "comparison",
        "Compare bases de dados SQL e NoSQL.": "comparison",
        "Que PFCs abordam segurança de aplicações web?": "general",
    }
    for query, expected in cases.items():
        actual = detect_query_intent(query)
        _assert(actual == expected, f"{query!r} -> {actual!r}, expected {expected!r}")


def test_hybrid_answerability_is_generic() -> None:
    retriever = object.__new__(HybridRetriever)
    generic = HybridRetriever._score_retrieval_answerability(
        retriever,
        text="Biometric authentication is defined as a method that verifies identity.",
        intent="definition",
    )
    portuguese = HybridRetriever._score_retrieval_answerability(
        retriever,
        text="Computação em nuvem é definido como um modelo de entrega de serviços.",
        intent="definition",
    )
    _assert(generic > 0.0, "English definition cue should score > 0.")
    _assert(portuguese > 0.0, "Portuguese definition cue should score > 0.")

    non_definition = HybridRetriever._score_retrieval_answerability(
        retriever,
        text="RAG addresses retrieval component external knowledge.",
        intent="comparison",
    )
    _assert(non_definition == 0.0, "Non-definition intent should not use definition boost.")


def test_original_query_always_present() -> None:
    rewriter = QueryRewriter(config=QueryRewriteConfig(use_llm=False, max_rewrites=3))
    for query in (
        "Define machine learning.",
        "Que PFCs abordam segurança de aplicações web?",
        "Compare SQL and NoSQL databases.",
    ):
        result = rewriter.rewrite(query)
        _assert(result.rewritten_queries[0] == query, "Original must be first when included.")
        _assert(
            result.rewrite_sources.get(query) == "original",
            f"Expected original rewrite source for {query!r}.",
        )


def test_acronym_expansion_only_when_present() -> None:
    rewriter = QueryRewriter(config=QueryRewriteConfig(use_llm=False))
    with_iot = rewriter.rewrite("Que PFCs existem sobre IoT?")
    joined = " | ".join(with_iot.rewritten_queries)
    _assert(re.search(r"Internet of Things", joined, re.IGNORECASE), "IoT should expand.")

    without = rewriter.rewrite("Que PFCs existem sobre redes?")
    joined_without = " | ".join(without.rewritten_queries).lower()
    _assert("internet of things" not in joined_without, "Must not invent IoT from networks.")


def main() -> int:
    test_production_sources_are_topic_neutral()
    test_intent_matrix()
    test_hybrid_answerability_is_generic()
    test_original_query_always_present()
    test_acronym_expansion_only_when_present()
    print("Slice 1 baseline regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
