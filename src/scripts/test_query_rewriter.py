import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure imports work when running from repository root:
# uv run python src/scripts/test_query_rewriter.py
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.query.intent import detect_query_intent
from rag.query.query_rewriter import QueryRewriteConfig, QueryRewriter
from rag.rerank.reranker import CrossEncoderReranker


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_no_rag_specific_production_heuristics() -> None:
    rewriter_source = Path(PROJECT_SRC, "rag", "query", "query_rewriter.py").read_text(
        encoding="utf-8"
    )
    hybrid_source = Path(PROJECT_SRC, "rag", "retrieval", "hybrid_retriever.py").read_text(
        encoding="utf-8"
    )
    rerank_source = Path(PROJECT_SRC, "rag", "rerank", "reranker.py").read_text(
        encoding="utf-8"
    )

    banned_phrases = (
        "rag addresses",
        "rag combines",
        "rag models augment",
        "retrieval component",
        "Retrieval-Augmented Generation is defined as",
        "hallucinations in RAG systems",
        "DEFAULT_DOMAIN_VOCABULARY",
    )
    combined = "\n".join([rewriter_source, hybrid_source, rerank_source]).lower()
    for phrase in banned_phrases:
        _assert(phrase.lower() not in combined, f"Found banned production phrase: {phrase}")


def test_intent_english_and_portuguese() -> None:
    cases = [
        ("Define machine learning.", "definition"),
        ("Defina computação em nuvem.", "definition"),
        ("What are the components of a recommendation system?", "components"),
        ("Quais são os componentes de uma arquitectura cliente-servidor?", "components"),
        ("How does biometric authentication work?", "mechanism"),
        ("Como funciona a autenticação biométrica?", "mechanism"),
        ("Compare SQL and NoSQL databases.", "comparison"),
        ("Compare bases de dados SQL e NoSQL.", "comparison"),
        ("Que PFCs abordam segurança de aplicações web?", "general"),
        ("What are hallucinations?", "definition"),
    ]
    for query, expected in cases:
        actual = detect_query_intent(query)
        _assert(actual == expected, f"Intent for '{query}' expected {expected}, got {actual}")


def test_rewrite_preserves_original_and_stays_topic_neutral() -> None:
    rewriter = QueryRewriter(config=QueryRewriteConfig(use_llm=False))

    result = rewriter.rewrite("Defina computação em nuvem.")
    _assert(result.original_query in result.rewritten_queries, "Original query must remain.")
    _assert(result.intent == "definition", "Expected definition intent.")
    joined = " | ".join(result.rewritten_queries).lower()
    for bad in ("rag", "hallucination", "medical", "psychiatric"):
        _assert(bad not in joined, f"Unexpected domain drift term '{bad}' in: {joined}")

    hallu = rewriter.rewrite("What are hallucinations?")
    joined_hallu = " | ".join(hallu.rewritten_queries).lower()
    _assert("hallucinations" in joined_hallu, "Subject should be preserved.")
    for invented in ("large language models", "rag systems", "medical", "psychiatric"):
        _assert(
            invented not in joined_hallu,
            f"Rule rewrite should not invent domain '{invented}'.",
        )

    security = rewriter.rewrite("Que PFCs abordam segurança de aplicações web?")
    joined_security = " | ".join(security.rewritten_queries).lower()
    _assert(
        "intrusion detection" not in joined_security
        and "deep learning" not in joined_security,
        "Must not invent a narrower cybersecurity subtopic.",
    )


def test_llm_rewrite_failure_falls_back_safely() -> None:
    failing_llm = MagicMock()
    failing_llm.invoke_messages.side_effect = RuntimeError("ollama unavailable")

    rewriter = QueryRewriter(
        llm_client=failing_llm,
        config=QueryRewriteConfig(use_llm=True, include_original=True),
    )
    result = rewriter.rewrite("Define machine learning.")

    _assert(result.used_llm is False, "LLM failure must not mark used_llm=True.")
    _assert(result.llm_error is not None, "LLM error should be preserved.")
    _assert(result.original_query in result.rewritten_queries, "Original must remain.")
    _assert(len(result.rewritten_queries) >= 1, "Fallback must still return queries.")


def test_validate_rejects_off_domain_and_requires_overlap() -> None:
    rewriter = QueryRewriter(config=QueryRewriteConfig(use_llm=False))
    original = "What are hallucinations?"

    _assert(
        rewriter._validate_llm_rewrite(original, "hallucinations definition overview"),
        "Overlapping rewrite should be accepted.",
    )
    _assert(
        not rewriter._validate_llm_rewrite(original, "hallucinations medical term"),
        "Medical hallucination rewrite should be rejected.",
    )
    _assert(
        not rewriter._validate_llm_rewrite(original, "hallucination psychiatric terminology"),
        "Psychiatric hallucination rewrite should be rejected.",
    )
    _assert(
        not rewriter._validate_llm_rewrite(original, "network intrusion detection deep learning"),
        "Unrelated domain rewrite without overlap should be rejected.",
    )


def test_generic_answerability_does_not_depend_on_rag() -> None:
    # Avoid loading the cross-encoder weights for this unit check.
    reranker = object.__new__(CrossEncoderReranker)
    chunk = (
        "Machine learning is defined as a field of artificial intelligence that "
        "enables systems to learn patterns from data."
    )
    score = CrossEncoderReranker.score_answerability(
        reranker,
        query="Define machine learning.",
        chunk=chunk,
        intent="definition",
        metadata={},
    )
    _assert(score > 0.0, "Generic definition phrasing should receive a positive score.")

    rag_only = "RAG addresses factuality by introducing a retrieval component."
    rag_score = CrossEncoderReranker.score_answerability(
        reranker,
        query="Define machine learning.",
        chunk=rag_only,
        intent="definition",
        metadata={},
    )
    _assert(
        rag_score <= score,
        "RAG-specific wording must not outrank generic definition phrasing by design.",
    )


def main() -> int:
    test_no_rag_specific_production_heuristics()
    test_intent_english_and_portuguese()
    test_rewrite_preserves_original_and_stays_topic_neutral()
    test_llm_rewrite_failure_falls_back_safely()
    test_validate_rejects_off_domain_and_requires_overlap()
    test_generic_answerability_does_not_depend_on_rag()
    print("All PFC-oriented QueryRewriter / intent regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
