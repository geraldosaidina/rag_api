"""Shared helpers for the Phase 1 retrieval benchmark runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.metrics.retrieval import mean_of
from rag.llm.ollama_client import OllamaLLMClient
from rag.query.query_rewriter import QueryRewriteConfig, QueryRewriter

CONFIGS: dict[str, dict[str, str]] = {
    "A1": {"retrieval_mode": "bm25", "rewrite_mode": "none"},
    "A2": {"retrieval_mode": "dense", "rewrite_mode": "none"},
    "A3": {"retrieval_mode": "hybrid", "rewrite_mode": "none"},
    "A4": {"retrieval_mode": "hybrid", "rewrite_mode": "rules"},
    "A5": {"retrieval_mode": "hybrid", "rewrite_mode": "rules_llm"},
}


@dataclass(frozen=True)
class GoldQuestion:
    question_id: str
    category: str
    question: str
    primary_pfc_ids: list[str]
    unsupported: bool


def load_gold(path: Path) -> tuple[dict[str, Any], list[GoldQuestion]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions: list[GoldQuestion] = []
    for item in payload["questions"]:
        expected = item["expected"]
        questions.append(
            GoldQuestion(
                question_id=str(item["id"]),
                category=str(item["category"]),
                question=str(item["question"]),
                primary_pfc_ids=[str(x) for x in expected.get("primary_pfc_ids") or []],
                unsupported=bool(expected.get("unsupported")),
            )
        )
    return payload, questions


def build_rewriter(
    rewrite_mode: str, llm_client: OllamaLLMClient | None
) -> QueryRewriter:
    if rewrite_mode == "none":
        return QueryRewriter(
            llm_client=None,
            config=QueryRewriteConfig(enabled=False, use_llm=False),
        )
    if rewrite_mode == "rules":
        return QueryRewriter(
            llm_client=None,
            config=QueryRewriteConfig(enabled=True, use_llm=False),
        )
    if rewrite_mode == "rules_llm":
        if llm_client is None:
            raise RuntimeError("rewrite_mode=rules_llm requires a working Ollama client.")
        return QueryRewriter(
            llm_client=llm_client,
            config=QueryRewriteConfig(enabled=True, use_llm=True),
        )
    raise ValueError(f"Unknown rewrite_mode: {rewrite_mode}")


def by_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["category"]), []).append(row)
    return {
        category: {
            "count": len(items),
            "precision_at_5": mean_of(item["precision_at_5"] for item in items),
            "recall_at_5": mean_of(item["recall_at_5"] for item in items),
            "recall_at_10": mean_of(item["recall_at_10"] for item in items),
            "mrr": mean_of(item["mrr"] for item in items),
        }
        for category, items in sorted(groups.items())
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if not row["unsupported"]]
    unsupported = [row for row in rows if row["unsupported"]]
    return {
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "supported_macro_averages": {
            "precision_at_5": mean_of(row["precision_at_5"] for row in supported),
            "recall_at_5": mean_of(row["recall_at_5"] for row in supported),
            "recall_at_10": mean_of(row["recall_at_10"] for row in supported),
            "mrr": mean_of(row["mrr"] for row in supported),
            "mean_retrieval_duration_ms": mean_of(
                row["retrieval_duration_ms"] for row in supported
            ),
        },
        "unsupported_question_ids": [row["question_id"] for row in unsupported],
        "unsupported_note": (
            "Unsupported queries are excluded from Recall and MRR averages "
            "(and from Precision@5 averages). Per-query rankings are still stored."
        ),
        "by_category_supported": by_category(supported),
    }
