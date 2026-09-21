"""Shared query-subject and acronym helpers for conservative rewriting."""

from __future__ import annotations

import re

COMMON_ACRONYM_EXPANSIONS = {
    "rag": "Retrieval-Augmented Generation",
    "llm": "Large Language Model",
    "llms": "Large Language Models",
    "iot": "Internet of Things",
    "nlp": "Natural Language Processing",
    "sql": "Structured Query Language",
    "nosql": "non-relational database",
    "api": "Application Programming Interface",
    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "pfc": "Projecto Final de Curso",
    "pfcs": "Projectos Finais de Curso",
}

_SUBJECT_PATTERNS = (
    r"^(?:what\s+are\s+the\s+(?:core\s+)?components\s+of)\s+(.+?)[\?\.]?$",
    r"^(?:quais\s+s[ãa]o\s+os\s+componentes\s+de(?:\s+uma|\s+um)?)\s+(.+?)[\?\.]?$",
    r"^(?:how\s+does(?:\s+the\s+author)?\s+define)\s+(.+?)[\?\.]?$",
    r"^(?:how\s+does)\s+(.+?)\s+work[\?\.]?$",
    r"^(?:how\s+do)\s+(.+?)\s+work[\?\.]?$",
    r"^(?:como\s+funciona(?:\s+a|\s+o)?)\s+(.+?)[\?\.]?$",
    r"^(?:how\s+does)\s+(.+?)[\?\.]?$",
    r"^(?:define|defina|definir)\s+(.+?)[\?\.]?$",
    r"^(?:what\s+(?:is|are)|o\s+que\s+(?:é|e|s[ãa]o))\s+(.+?)[\?\.]?$",
    r"^(?:compare|comparar)\s+(.+?)[\?\.]?$",
    r"^(?:que\s+pfcs?\s+(?:existem\s+sobre|abordam|falam\s+de))\s+(.+?)[\?\.]?$",
    r"^(?:existem\s+trabalhos?\s+(?:relacionados?\s+com|sobre))\s+(.+?)[\?\.]?$",
)


def extract_subject(query: str) -> str:
    text = query.strip()
    lower = text.lower()
    for pattern in _SUBJECT_PATTERNS:
        match = re.match(pattern, lower, flags=re.IGNORECASE)
        if match:
            start, end = match.start(1), match.end(1)
            return text[start:end].strip(" ?.!,;:")
    cleaned = re.sub(r"^(?:please\s+)?(?:can\s+you\s+)?", "", text, flags=re.IGNORECASE)
    return cleaned.strip(" ?.!,;:")


def acronym_expansions(query: str) -> list[str]:
    expansions: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", query):
        expansion = COMMON_ACRONYM_EXPANSIONS.get(token.lower())
        if expansion and expansion.lower() not in query.lower():
            expansions.append(f"{token} {expansion}")
    return expansions


def intent_rule_rewrites(subject: str, intent: str) -> list[str]:
    if not subject:
        return []
    subject_lower = subject.lower()
    if intent == "mechanism" and subject_lower.startswith(("how ", "como ")):
        return []
    templates = {
        "definition": [
            f"{subject} is defined as",
            f"{subject} refers to",
            f"{subject} é definido como",
            f"{subject} refere-se a",
        ],
        "components": [
            f"components of {subject}",
            f"{subject} consists of",
            f"componentes de {subject}",
            f"{subject} é composto por",
        ],
        "mechanism": [
            f"how {subject} works",
            f"{subject} works by",
            f"como funciona {subject}",
            f"funcionamento de {subject}",
        ],
        "comparison": [
            f"difference between {subject}",
            f"comparison of {subject}",
            f"diferença entre {subject}",
            f"comparação entre {subject}",
        ],
    }
    return list(templates.get(intent, []))
