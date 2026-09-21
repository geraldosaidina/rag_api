"""Generic answerability cues for definition-style retrieval and reranking."""

from __future__ import annotations

import re
from typing import Any


DEFINITION_POSITIVE_PHRASES: tuple[tuple[str, float], ...] = (
    ("is defined as", 0.12),
    ("can be defined as", 0.12),
    ("refers to", 0.10),
    ("is a method that", 0.09),
    ("is a framework that", 0.09),
    ("é definido como", 0.12),
    ("é definida como", 0.12),
    ("refere-se a", 0.10),
    ("é um método que", 0.09),
    ("é um metodo que", 0.09),
    ("consiste em", 0.06),
    ("by integrating", 0.06),
    ("by combining", 0.06),
)

RETRIEVAL_DEFINITION_PHRASES: tuple[tuple[str, float], ...] = (
    ("is defined as", 0.04),
    ("can be defined as", 0.04),
    ("refers to", 0.03),
    ("is a method", 0.03),
    ("is a framework", 0.03),
    ("consists of", 0.02),
    ("é definido como", 0.04),
    ("é definida como", 0.04),
    ("refere-se a", 0.03),
    ("é um método", 0.03),
    ("é um metodo", 0.03),
    ("é uma framework", 0.03),
    ("consiste em", 0.02),
)

WEAK_ACADEMIC_PREFIXES: tuple[tuple[str, float], ...] = (
    ("this paper", 0.12),
    ("in this survey", 0.12),
    ("we discuss", 0.10),
    ("open issues", 0.20),
    ("future directions", 0.18),
    ("neste trabalho", 0.12),
    ("neste artigo", 0.12),
    ("neste survey", 0.12),
)


def score_definition_phrases(
    text: str,
    phrase_weights: tuple[tuple[str, float], ...],
    *,
    cap: float,
) -> float:
    lower = text.lower()
    score = 0.0
    for phrase, value in phrase_weights:
        if phrase in lower:
            score += value
    return min(max(score, 0.0), cap)


def explanation_sentence_bonus(chunk: str) -> float:
    for sentence in re.split(r"(?<=[.!?])\s+", chunk):
        word_count = len(re.findall(r"[A-Za-z0-9À-ÿ]+", sentence))
        if word_count < 14:
            continue
        if re.search(
            r"\b(is|are|means|uses|integrates|combines|provides|enables|"
            r"é|sao|são|significa|utiliza|combina|integra|permite)\b",
            sentence.lower(),
        ):
            return 0.05
    return 0.0


def definition_penalty(chunk: str, metadata: dict[str, Any] | None = None) -> float:
    lower = chunk.lower().lstrip()
    penalty = 0.0
    for prefix, value in WEAK_ACADEMIC_PREFIXES:
        if lower.startswith(prefix):
            penalty += value
            break
    if str((metadata or {}).get("content_type", "normal")).lower() == "mixed":
        penalty += 0.10
    return min(penalty, 0.30)


def rerank_definition_answerability(
    chunk: str,
    metadata: dict[str, Any] | None = None,
) -> float:
    if not chunk or not chunk.strip():
        return 0.0
    positive = score_definition_phrases(chunk, DEFINITION_POSITIVE_PHRASES, cap=0.25)
    positive += explanation_sentence_bonus(chunk)
    positive = min(positive, 0.25)
    score = positive - definition_penalty(chunk, metadata)
    return max(min(score, 0.25), -0.30)


def retrieval_definition_boost(text: str) -> float:
    return score_definition_phrases(text, RETRIEVAL_DEFINITION_PHRASES, cap=0.15)


def adjust_definition_candidate_score(content: str, score: float) -> float:
    lower = content.lower().strip()
    weak_prefixes = (
        "this paper concludes",
        "open issues",
        "future directions",
        "neste trabalho",
        "este artigo",
    )
    if any(lower.startswith(prefix) for prefix in weak_prefixes):
        score -= 0.08
    score += min(retrieval_definition_boost(content) * 2.5, 0.12)
    return max(score, 0.0)
