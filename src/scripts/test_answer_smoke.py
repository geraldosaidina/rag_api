"""End-to-end smoke test for Slice 2 grounded QA.

Run from repository root:
  uv run python src/scripts/test_answer_smoke.py

Requires:
  - indexed Chroma collection pfc_corpus
  - Ollama with llama3.1:8b
  - local embedding/reranker model cache
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.service.factory import RAGAppConfig, build_rag_service
from rag.service.models import AnswerResult
from rag.service.rag_service import RAGServiceException


def _print_result(result: AnswerResult) -> None:
    print("\n" + "=" * 100)
    print(f"QUESTION: {result.question}")
    print("-" * 100)
    print("ANSWER:")
    print(result.answer)
    print("-" * 100)
    print("SOURCES:")
    if not result.sources:
        print("(none)")
    else:
        for source in result.sources:
            print(
                f"- {source.citation_id} | source={source.source} | "
                f"page={source.page} | chunk_index={source.chunk_index} | "
                f"chunk_id={source.chunk_id}"
            )
            if source.excerpt:
                print(f"  excerpt: {source.excerpt}")
    print("-" * 100)
    print("DIAGNOSTICS:")
    print(
        json.dumps(
            {
                "total_duration_ms": round(result.diagnostics.total_duration_ms, 1),
                "retrieval_duration_ms": (
                    None
                    if result.diagnostics.retrieval_duration_ms is None
                    else round(result.diagnostics.retrieval_duration_ms, 1)
                ),
                "rerank_duration_ms": (
                    None
                    if result.diagnostics.rerank_duration_ms is None
                    else round(result.diagnostics.rerank_duration_ms, 1)
                ),
                "generation_duration_ms": (
                    None
                    if result.diagnostics.generation_duration_ms is None
                    else round(result.diagnostics.generation_duration_ms, 1)
                ),
                "retrieval_candidate_count": result.diagnostics.retrieval_candidate_count,
                "evidence_chunk_count": result.diagnostics.evidence_chunk_count,
                "used_llm_query_rewrite": result.diagnostics.used_llm_query_rewrite,
                "insufficient_evidence": result.diagnostics.insufficient_evidence,
                "invalid_citations_removed": list(
                    result.diagnostics.invalid_citations_removed
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    questions = [
        "Como funciona Retrieval-Augmented Generation?",
        "Quais são os principais componentes de um sistema RAG?",
        "Como RAG pode reduzir alucinações?",
        "Qual é a capital da Austrália segundo os PFCs indexados?",
    ]

    print("Building reusable RAG stack once...")
    try:
        service = build_rag_service(
            RAGAppConfig(
                candidate_k=40,
                rerank_top_k=8,
                max_evidence_chunks=5,
            )
        )
    except Exception as exc:
        print(f"Failed to build RAG stack: {exc}")
        return 1

    success = 0
    for question in questions:
        try:
            result = service.answer(question)
            _print_result(result)
            success += 1
        except RAGServiceException as exc:
            print("\n" + "=" * 100)
            print(f"QUESTION: {question}")
            print(f"INFRASTRUCTURE/APPLICATION ERROR: {exc}")
        except Exception as exc:
            print("\n" + "=" * 100)
            print(f"QUESTION: {question}")
            print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
            return 1

    print("\n" + "=" * 100)
    print(f"Completed {success}/{len(questions)} questions.")
    return 0 if success == len(questions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
