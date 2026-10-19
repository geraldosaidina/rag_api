"""Phase 1 retrieval benchmark (configurations A1–A5).

Runs offline against evaluation/gold_questions.json. Does not call the student
API, does not generate answers, and does not reingest the corpus.

Example:
  uv run python evaluation/run_benchmark.py
  uv run python evaluation/run_benchmark.py --configs A1,A2,A3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
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

from evaluation.benchmark_support import (  # noqa: E402
    CONFIGS,
    aggregate,
    build_rewriter,
    load_gold,
)
from evaluation.metrics.retrieval import (  # noqa: E402
    collapse_to_document_ranking,
    score_ranking,
)
from rag.embeddings.embedder import EmbeddingConfig, build_embeddings_client  # noqa: E402
from rag.llm.ollama_client import LLMConfig, OllamaLLMClient  # noqa: E402
from rag.query.query_rewriter import QueryRewriteConfig, QueryRewriter  # noqa: E402
from rag.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from rag.service.factory import load_rag_app_config  # noqa: E402
from rag.vector.chroma_store import ChromaConfig, ChromaVectorStore  # noqa: E402

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


def run_benchmark(
    *,
    gold_path: Path,
    config_ids: list[str],
    candidate_k: int,
    results_root: Path,
) -> Path:
    gold_payload, questions = load_gold(gold_path)
    app_config = load_rag_app_config()
    print(
        f"Loading embeddings ({app_config.embedding_model_name}) and Chroma "
        f"collection '{app_config.collection_name}'...",
        flush=True,
    )
    embedding_fn = build_embeddings_client(
        EmbeddingConfig(model_name=app_config.embedding_model_name)
    )
    vector_store = ChromaVectorStore(
        embedding_function=embedding_fn,
        config=ChromaConfig(
            persist_directory=app_config.persist_directory,
            collection_name=app_config.collection_name,
        ),
    )
    chunk_count = vector_store.count()
    print(f"Corpus chunks: {chunk_count}", flush=True)

    llm_client: OllamaLLMClient | None = None
    if any(CONFIGS[cid]["rewrite_mode"] == "rules_llm" for cid in config_ids):
        print(
            f"Preparing Ollama client for rewrite_mode=rules_llm "
            f"({app_config.llm_model_name})...",
            flush=True,
        )
        llm_client = OllamaLLMClient(
            LLMConfig(
                model_name=app_config.llm_model_name,
                base_url=app_config.ollama_base_url,
            )
        )
        if not llm_client.health_check():
            raise RuntimeError(
                "rewrite_mode=rules_llm requires a reachable Ollama model "
                f"'{app_config.llm_model_name}'. Start Ollama and retry. "
                "Refusing to degrade silently to rules-only."
            )
        print("Ollama health check passed.", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    retrieval_path = run_dir / "retrieval.jsonl"

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "1_retrieval",
        "dataset_path": str(gold_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_version": gold_payload.get("dataset_version"),
        "corpus_collection": app_config.collection_name,
        "corpus_persist_directory": app_config.persist_directory,
        "corpus_chunk_count": chunk_count,
        "embedding_model": app_config.embedding_model_name,
        "llm_model": app_config.llm_model_name,
        "candidate_k": candidate_k,
        "chunk_size": 400,
        "chunk_overlap": 100,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "configs": {cid: CONFIGS[cid] for cid in config_ids},
        "relevance_protocol": {
            "unit": "document (pfc_id)",
            "relevant_field": "expected.primary_pfc_ids",
            "secondary_pfc_ids": "ignored (empty in gold v1.0)",
            "collapse": "first chunk occurrence of each pfc_id",
            "unsupported_excluded_from_macro_recall_mrr_precision": True,
        },
        "annotation_limitations": [
            "Gold relevance is document-level, not chunk-level.",
            "secondary_pfc_ids are empty for all questions in gold v1.0.",
            "evidence.pdf_pages are not used for Phase 1 IR metrics.",
            "answer_notes are factual notes, not canonical answers.",
            "Unsupported queries have an empty relevant set; Recall/MRR are undefined.",
        ],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Load Chroma/BM25 once. Only the rewriter and retrieval_mode change per config.
    print("Building BM25 index from Chroma documents...", flush=True)
    base_retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_function=embedding_fn,
        query_rewriter=QueryRewriter(
            config=QueryRewriteConfig(enabled=False, use_llm=False)
        ),
    )
    print(f"BM25 documents loaded: {len(base_retriever._documents)}", flush=True)

    summary_configs: dict[str, Any] = {}
    with retrieval_path.open("w", encoding="utf-8") as handle:
        for config_id in config_ids:
            spec = CONFIGS[config_id]
            base_retriever.query_rewriter = build_rewriter(
                spec["rewrite_mode"], llm_client
            )
            rows: list[dict[str, Any]] = []
            print(
                f"\n=== {config_id}: mode={spec['retrieval_mode']} "
                f"rewrite={spec['rewrite_mode']}",
                flush=True,
            )
            for question in questions:
                started = time.perf_counter()
                hits = base_retriever.retrieve(
                    query=question.question,
                    candidate_k=candidate_k,
                    retrieval_mode=spec["retrieval_mode"],
                )
                duration_ms = (time.perf_counter() - started) * 1000.0
                ranked_docs = collapse_to_document_ranking(hits)
                metrics = score_ranking(
                    ranked_docs,
                    question.primary_pfc_ids,
                    unsupported=question.unsupported,
                )
                rewrite_meta = hits[0].metadata if hits else {}
                used_llm = bool(rewrite_meta.get("used_llm_query_rewrite"))
                llm_error = rewrite_meta.get("llm_error")
                if spec["rewrite_mode"] == "rules_llm" and llm_error:
                    raise RuntimeError(
                        f"{config_id}/{question.question_id}: rewrite_mode=rules_llm "
                        "failed to invoke the LLM. Aborting rather than silently "
                        f"degrading to rules-only. llm_error={llm_error!r}"
                    )
                row = {
                    "run_id": run_id,
                    "config_id": config_id,
                    "retrieval_mode": spec["retrieval_mode"],
                    "rewrite_mode": spec["rewrite_mode"],
                    "question_id": question.question_id,
                    "category": question.category,
                    "unsupported": question.unsupported,
                    "question": question.question,
                    "primary_relevant": question.primary_pfc_ids,
                    "candidate_k": candidate_k,
                    "ranked_pfc_ids": ranked_docs,
                    "ranked_chunk_ids": [
                        (hit.metadata or {}).get("chunk_id") for hit in hits
                    ],
                    "rewritten_queries": list(
                        rewrite_meta.get("rewritten_queries")
                        or [question.question]
                    ),
                    "used_llm_query_rewrite": used_llm,
                    "llm_error": llm_error,
                    "retrieval_duration_ms": duration_ms,
                    **metrics,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                print(
                    f"  {question.question_id} unsupported={question.unsupported} "
                    f"P@5={metrics['precision_at_5']} R@5={metrics['recall_at_5']} "
                    f"R@10={metrics['recall_at_10']} MRR={metrics['mrr']} "
                    f"ms={duration_ms:.0f}",
                    flush=True,
                )
            summary_configs[config_id] = {
                "spec": spec,
                **aggregate(rows),
            }

    summary = {
        "run_id": run_id,
        "dataset_version": gold_payload.get("dataset_version"),
        "corpus_collection": app_config.collection_name,
        "corpus_chunk_count": chunk_count,
        "configs": summary_configs,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote results to {run_dir}", flush=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 retrieval benchmark (A1–A5).")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
        help="Path to gold_questions.json",
    )
    parser.add_argument(
        "--configs",
        default="A1,A2,A3,A4,A5",
        help="Comma-separated config ids (default: A1-A5)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=int(os.getenv("RAG_CANDIDATE_K", "40")),
        help="Candidate pool size (default: production RAG_CANDIDATE_K or 40)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Directory for run outputs",
    )
    args = parser.parse_args()
    config_ids = [item.strip() for item in args.configs.split(",") if item.strip()]
    unknown = [item for item in config_ids if item not in CONFIGS]
    if unknown:
        print(f"Unknown config ids: {unknown}. Allowed: {sorted(CONFIGS)}", file=sys.stderr)
        return 2
    if args.candidate_k <= 0:
        print("--candidate-k must be > 0", file=sys.stderr)
        return 2
    if not args.gold.is_file():
        print(f"Gold dataset not found: {args.gold}", file=sys.stderr)
        return 2

    run_benchmark(
        gold_path=args.gold,
        config_ids=config_ids,
        candidate_k=args.candidate_k,
        results_root=args.results_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
