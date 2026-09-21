"""Lightweight query-intent heuristics shared by rewrite, retrieval, and rerank."""

from __future__ import annotations

import re

Intent = str  # definition | comparison | mechanism | components | general


def detect_query_intent(query: str) -> Intent:
    """
    Detect a coarse retrieval intent from surface cues.

    Priority: components > comparison > strong definition > mechanism > soft definition > general.
    Supports common English and Portuguese patterns without a full NLP stack.
    """
    lower = query.lower()

    components_terms = (
        "components",
        "parts",
        "elements",
        "architecture",
        "modules",
        "componentes",
        "partes",
        "elementos",
        "arquitectura",
        "arquitetura",
        "módulos",
        "modulos",
    )
    comparison_terms = (
        "compare",
        "comparison",
        "difference",
        "differences",
        "differentiate",
        "versus",
        " vs ",
        " vs.",
        "different from",
        "differ from",
        "comparar",
        "compare ",
        "diferença",
        "diferenca",
        "diferenças",
        "diferencas",
        "diferente de",
        "versus",
    )
    mechanism_terms = (
        "how does",
        "how do",
        "how can",
        "how is",
        "works",
        "work?",
        "reduce",
        "improve",
        "affect",
        "como funciona",
        "como é que",
        "como e que",
        "de que forma",
        "de que maneira",
    )
    strong_definition_terms = (
        "define",
        "definition",
        "meaning of",
        "refers to",
        "defina",
        "definir",
        "definição",
        "definicao",
        "significa",
        "refere-se",
        "refere se",
    )
    soft_definition_terms = (
        "what is",
        "what are",
        "what does",
        "o que é",
        "o que e",
        "o que são",
        "o que sao",
        "que é",
        "que e",
    )

    if any(term in lower for term in components_terms):
        return "components"
    if any(term in lower for term in comparison_terms) or re.search(r"\bvs\.?\b", lower):
        return "comparison"
    if any(term in lower for term in strong_definition_terms):
        return "definition"
    if any(term in lower for term in mechanism_terms):
        return "mechanism"
    if any(term in lower for term in soft_definition_terms):
        return "definition"
    return "general"
