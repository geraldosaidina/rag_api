"""Phase 2 generation benchmark (production stack, H1/H4).

Runs full RAGService.answer() over gold_questions.json with production defaults.
Stores evidence traces, automatic citation metrics, and optional LLM judgements.

Example:
  uv run python evaluation/run_generation_benchmark.py
  uv run python evaluation/run_generation_benchmark.py --skip-llm-judge
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.benchmark_support import load_gold  # noqa: E402
from evaluation.generation_support import (  # noqa: E402
    aggregate_generation,
    answer_notes,
    gold_pages,
    serialize_evidence,
    serialize_sources,
)
from evaluation.judging.llm_judge import (  # noqa: E402
    JUDGE_SCALE,
    SYSTEM_PROMPT,
    judge_answer,
    judge_prompt_record,
)
from evaluation.metrics.generation import abstention_signals, score_citations  # noqa: E402
from rag.llm.ollama_client import LLMConfig, OllamaLLMClient  # noqa: E402
from rag.service.factory import build_rag_service, load_rag_app_config  # noqa: E402
from rag.service.rag_service import RAGServiceException  # noqa: E402

DEFAULT_GOLD = Path(__file__).resolve().parent / "gold_questions.json"
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "runs"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _apply_auto_human_flags(judge_payload: dict[str, Any], citation_metrics: dict[str, Any]) -> None:
    reasons = list(judge_payload.get("human_review_reasons") or [])
    if citation_metrics.get("citation_doc_precision") not in (None, 1.0):
        judge_payload["needs_human_review"] = True
        reasons.append("citation_doc_precision_below_one")
    if citation_metrics["invalid_citation_count"] > 0:
        judge_payload["needs_human_review"] = True
        reasons.append("invalid_citations_removed")
    judge_payload["human_review_reasons"] = reasons


def run_generation_benchmark(
    *,
    gold_path: Path,
    results_root: Path,
    skip_llm_judge: bool,
) -> Path:
    gold_payload, questions = load_gold(gold_path)
    pages_by_question = gold_pages(gold_payload)
    notes_by_question = answer_notes(gold_payload)
    app_config = load_rag_app_config()

    print("Building production RAG stack...", flush=True)
    service = build_rag_service(app_config)
    chunk_count = service.retriever.vector_store.count()

    judge_client: OllamaLLMClient | None = None
    if not skip_llm_judge:
        judge_client = OllamaLLMClient(
            LLMConfig(
                model_name=app_config.llm_model_name,
                base_url=app_config.ollama_base_url,
            )
        )
        if not judge_client.health_check():
            raise RuntimeError(
                "LLM judge requires a reachable Ollama model "
                f"'{app_config.llm_model_name}'. Start Ollama or pass --skip-llm-judge."
            )
        print("Ollama health check passed for generation + judge.", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    generation_path = run_dir / "generation.jsonl"
    judgments_path = run_dir / "judgments.jsonl"

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "2_generation",
        "config_id": "B1",
        "config": {
            "retrieval_mode": "hybrid",
            "rewrite_mode": "rules_llm",
            "rerank": True,
            "generation": True,
        },
        "dataset_path": str(gold_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_version": gold_payload.get("dataset_version"),
        "corpus_collection": app_config.collection_name,
        "corpus_chunk_count": chunk_count,
        "embedding_model": app_config.embedding_model_name,
        "reranker_model": app_config.reranker_model_name,
        "llm_model": app_config.llm_model_name,
        "candidate_k": app_config.candidate_k,
        "rerank_top_k": app_config.rerank_top_k,
        "max_evidence_chunks": app_config.max_evidence_chunks,
        "max_evidence_chars": app_config.max_evidence_chars,
        "use_llm_query_rewrite": app_config.use_llm_query_rewrite,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "llm_judge_enabled": not skip_llm_judge,
        "llm_judge": None if skip_llm_judge else judge_prompt_record(app_config.llm_model_name),
        "protocol_path": "evaluation/judging/protocol.md",
        "notes": [
            "Gold answer_notes are judge hints only, not canonical answers.",
            "Student API/frontend unchanged; results stored under evaluation/results/.",
        ],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    with generation_path.open("w", encoding="utf-8") as gen_handle, judgments_path.open(
        "w", encoding="utf-8"
    ) as judge_handle:
        for question in questions:
            print(f"\n=== {question.question_id} unsupported={question.unsupported}", flush=True)
            try:
                result = service.answer(question.question)
            except RAGServiceException as exc:
                raise RuntimeError(f"Pipeline failed on {question.question_id}: {exc}") from exc

            sources = serialize_sources(result)
            evidence_items = serialize_evidence(result)
            citation_metrics = score_citations(
                sources=sources,
                primary_pfc_ids=question.primary_pfc_ids,
                gold_pages_by_pfc=pages_by_question.get(question.question_id, {}),
                unsupported=question.unsupported,
                invalid_citations_removed=result.diagnostics.invalid_citations_removed,
            )
            abstention = abstention_signals(
                answer=result.answer,
                insufficient_evidence=result.diagnostics.insufficient_evidence,
                unsupported=question.unsupported,
            )
            row: dict[str, Any] = {
                "run_id": run_id,
                "config_id": "B1",
                "question_id": question.question_id,
                "category": question.category,
                "unsupported": question.unsupported,
                "question": question.question,
                "primary_relevant": question.primary_pfc_ids,
                "answer": result.answer,
                "raw_answer_before_citation_cleanup": result.raw_answer_before_citation_cleanup,
                "evidence_context_text": result.evidence_context_text,
                "evidence_items": evidence_items,
                "sources": sources,
                "insufficient_evidence": result.diagnostics.insufficient_evidence,
                "used_llm_query_rewrite": result.diagnostics.used_llm_query_rewrite,
                "retrieval_candidate_count": result.diagnostics.retrieval_candidate_count,
                "evidence_chunk_count": result.diagnostics.evidence_chunk_count,
                "retrieval_duration_ms": result.diagnostics.retrieval_duration_ms,
                "rerank_duration_ms": result.diagnostics.rerank_duration_ms,
                "generation_duration_ms": result.diagnostics.generation_duration_ms,
                "total_duration_ms": result.diagnostics.total_duration_ms,
                "citation_metrics": citation_metrics,
                "abstention": abstention,
                "llm_judge": None,
            }

            if judge_client is not None:
                judged = judge_answer(
                    judge_client,
                    question=question.question,
                    answer=result.answer,
                    evidence_context_text=result.evidence_context_text or "",
                    sources=sources,
                    unsupported=question.unsupported,
                    answer_notes=notes_by_question.get(question.question_id) or None,
                )
                judge_payload = {
                    "question_id": question.question_id,
                    "model_name": judged.model_name,
                    "system_prompt": SYSTEM_PROMPT,
                    "scale": JUDGE_SCALE,
                    "faithfulness": judged.faithfulness,
                    "answer_relevance": judged.answer_relevance,
                    "citation_semantic_support": judged.citation_semantic_support,
                    "unsupported_handling": judged.unsupported_handling,
                    "rationale": judged.rationale,
                    "needs_human_review": judged.needs_human_review,
                    "human_review_reasons": judged.human_review_reasons,
                    "parse_error": judged.parse_error,
                    "raw_response": judged.raw_response,
                }
                _apply_auto_human_flags(judge_payload, citation_metrics)
                judge_handle.write(json.dumps(judge_payload, ensure_ascii=False) + "\n")
                judge_handle.flush()
                row["llm_judge"] = {
                    key: judge_payload[key]
                    for key in (
                        "model_name",
                        "faithfulness",
                        "answer_relevance",
                        "citation_semantic_support",
                        "unsupported_handling",
                        "rationale",
                        "needs_human_review",
                        "human_review_reasons",
                        "parse_error",
                    )
                }
                print(
                    f"  judge F={judged.faithfulness} R={judged.answer_relevance} "
                    f"C={judged.citation_semantic_support} U={judged.unsupported_handling} "
                    f"human={judge_payload['needs_human_review']}",
                    flush=True,
                )

            gen_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            gen_handle.flush()
            rows.append(row)
            print(
                f"  sources={len(sources)} insufficient={row['insufficient_evidence']} "
                f"docP={citation_metrics['citation_doc_precision']} "
                f"pageP={citation_metrics['citation_page_precision']} "
                f"total_ms={row['total_duration_ms']:.0f}",
                flush=True,
            )

    summary = {
        "run_id": run_id,
        "phase": "2_generation",
        "dataset_version": gold_payload.get("dataset_version"),
        "corpus_collection": app_config.collection_name,
        "corpus_chunk_count": chunk_count,
        "llm_judge_enabled": not skip_llm_judge,
        **aggregate_generation(rows),
        "limitations": [
            "Automatic citation metrics use gold primary_pfc_ids and pdf_pages only.",
            "Faithfulness/relevance LLM scores require human review for mid/failed cases.",
            "answer_notes are not canonical answers.",
            "H1/H4 are not automatically confirmed by these scores.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote results to {run_dir}", flush=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 generation benchmark (B1).")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Store generation traces and automatic citation metrics only.",
    )
    args = parser.parse_args()
    if not args.gold.is_file():
        print(f"Gold dataset not found: {args.gold}", file=sys.stderr)
        return 2
    run_generation_benchmark(
        gold_path=args.gold,
        results_root=args.results_root,
        skip_llm_judge=args.skip_llm_judge,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
