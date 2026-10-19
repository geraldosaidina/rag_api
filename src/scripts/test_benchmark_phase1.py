"""Non-regression checks for Phase 1 benchmark hooks.

Confirms production defaults:
- retrieve() without retrieval_mode behaves as hybrid
- QueryRewriteConfig(enabled=True, use_llm=False) still applies rules
- enabled=False disables all rewrites (benchmark rewrite_none only)

Run:
  uv run python src/scripts/test_benchmark_phase1.py
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.query.query_rewriter import QueryRewriteConfig, QueryRewriter
from rag.retrieval.hybrid_retriever import RETRIEVAL_MODES, HybridRetriever
from rag.service.factory import load_rag_app_config
from rag.service.rag_service import RAGService
from rag.vector.chroma_store import SearchResult


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeVectorStore:
    def __init__(self, hits: list[SearchResult]):
        self._hits = hits
        self.dense_calls = 0

    def similarity_search_with_scores(self, query: str, k: int) -> list[SearchResult]:
        self.dense_calls += 1
        return list(self._hits)[:k]

    @property
    def _client(self) -> Any:
        return self


class _FakeClient:
    def __init__(self, documents: list[str], metadatas: list[dict[str, Any]]):
        self._documents = documents
        self._metadatas = metadatas

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        _ = include
        return {"documents": self._documents, "metadatas": self._metadatas}


def _make_retriever(
    *,
    dense_hits: list[SearchResult],
    bm25_docs: list[dict[str, Any]],
    rewriter: QueryRewriter | None,
) -> HybridRetriever:
    store = _FakeVectorStore(dense_hits)
    # HybridRetriever loads BM25 from Chroma in __init__; inject fakes after.
    retriever = object.__new__(HybridRetriever)
    retriever.vector_store = store
    retriever.embedding_function = object()
    retriever.query_rewriter = rewriter
    retriever._documents = bm25_docs
    from rag.retrieval.bm25_index import Bm25DocumentIndex

    retriever._bm25_index = Bm25DocumentIndex(bm25_docs)
    return retriever


def test_retrieve_signature_default_is_hybrid() -> None:
    signature = inspect.signature(HybridRetriever.retrieve)
    mode_param = signature.parameters["retrieval_mode"]
    _assert(mode_param.default == "hybrid", "Default retrieval_mode must remain hybrid.")
    _assert(RETRIEVAL_MODES == frozenset({"hybrid", "dense", "bm25"}), "Unexpected modes.")


def test_rag_service_calls_retrieve_without_mode_kwarg() -> None:
    source = Path(PROJECT_SRC / "rag" / "service" / "rag_service.py").read_text(
        encoding="utf-8"
    )
    _assert(
        "self.retriever.retrieve(" in source,
        "RAGService must still call retriever.retrieve.",
    )
    # Production call site must not force a non-default mode.
    _assert(
        "retrieval_mode=" not in source,
        "RAGService must not pass retrieval_mode (default hybrid).",
    )
    _assert(
        "candidate_k=self.config.candidate_k" in source,
        "Production candidate_k wiring changed unexpectedly.",
    )


def test_rewrite_none_disables_rules_and_llm() -> None:
    rewriter = QueryRewriter(
        llm_client=None,
        config=QueryRewriteConfig(enabled=False, use_llm=False),
    )
    result = rewriter.rewrite("Que PFCs existem sobre IoT?")
    _assert(result.rewritten_queries == ["Que PFCs existem sobre IoT?"], "Only original.")
    _assert(result.used_llm is False, "LLM must not run when enabled=False.")
    joined = " | ".join(result.rewritten_queries)
    _assert(
        not re.search(r"Internet of Things", joined, re.IGNORECASE),
        "Rules must not expand acronyms when enabled=False.",
    )


def test_rewrite_rules_still_expand_when_enabled() -> None:
    rewriter = QueryRewriter(
        llm_client=None,
        config=QueryRewriteConfig(enabled=True, use_llm=False),
    )
    result = rewriter.rewrite("Que PFCs existem sobre IoT?")
    joined = " | ".join(result.rewritten_queries)
    _assert(result.rewritten_queries[0].startswith("Que PFCs"), "Original first.")
    _assert(
        re.search(r"Internet of Things", joined, re.IGNORECASE),
        "Production-compatible rules mode must still expand IoT.",
    )


def test_factory_defaults_keep_rewrite_enabled() -> None:
    cfg = load_rag_app_config()
    # Default env: LLM rewrite on. enabled flag is not exposed via env and stays True.
    _assert(cfg.use_llm_query_rewrite is True, "Default use_llm_query_rewrite should be True.")
    _assert(cfg.candidate_k == 40, "Default candidate_k should remain 40.")
    _assert(cfg.collection_name == "pfc_corpus", "Default collection should remain pfc_corpus.")


def test_dense_and_bm25_modes_skip_unused_channel() -> None:
    dense_hit = SearchResult(
        content="dense only passage about fibre access",
        metadata={
            "pfc_id": "dense-doc",
            "chunk_id": "dense-doc:page-1:chunk-0:aaa",
            "section_type": "body",
            "content_type": "normal",
            "retrieval_quality": "normal",
        },
        score=0.2,
    )
    bm25_docs = [
        {
            "content": "bm25 only passage about wireless coverage marracuene",
            "metadata": {
                "pfc_id": "bm25-doc",
                "chunk_id": "bm25-doc:page-1:chunk-0:bbb",
                "section_type": "body",
                "content_type": "normal",
                "retrieval_quality": "normal",
            },
        }
    ]
    rewriter = QueryRewriter(config=QueryRewriteConfig(enabled=False, use_llm=False))
    retriever = _make_retriever(
        dense_hits=[dense_hit],
        bm25_docs=bm25_docs,
        rewriter=rewriter,
    )

    dense_results = retriever.retrieve("wireless fibre", candidate_k=5, retrieval_mode="dense")
    _assert(retriever.vector_store.dense_calls == 1, "Dense mode must call vector search.")
    _assert(len(dense_results) >= 1, "Dense mode should return dense hits.")
    _assert(
        dense_results[0].metadata.get("pfc_id") == "dense-doc",
        "Dense mode should prefer dense channel.",
    )
    _assert(dense_results[0].metadata.get("retrieval_mode") == "dense", "Mode metadata.")

    retriever.vector_store.dense_calls = 0
    bm25_results = retriever.retrieve(
        "wireless coverage marracuene",
        candidate_k=5,
        retrieval_mode="bm25",
    )
    _assert(retriever.vector_store.dense_calls == 0, "BM25 mode must not call dense search.")
    _assert(len(bm25_results) >= 1, "BM25 mode should return sparse hits.")
    _assert(
        bm25_results[0].metadata.get("pfc_id") == "bm25-doc",
        "BM25 mode should return BM25 document.",
    )

    hybrid_results = retriever.retrieve(
        "wireless fibre coverage",
        candidate_k=5,
        retrieval_mode="hybrid",
    )
    _assert(retriever.vector_store.dense_calls == 1, "Hybrid mode calls dense search.")
    _assert(hybrid_results[0].metadata.get("retrieval_mode") == "hybrid", "Hybrid metadata.")


def test_metrics_helpers() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from evaluation.metrics.retrieval import (
        collapse_to_document_ranking,
        mean_reciprocal_rank,
        precision_at_k,
        recall_at_k,
        score_ranking,
    )

    class Hit:
        def __init__(self, pfc_id: str):
            self.metadata = {"pfc_id": pfc_id}

    ranked = collapse_to_document_ranking([Hit("a"), Hit("a"), Hit("b"), Hit("c")])
    _assert(ranked == ["a", "b", "c"], "Collapse must drop duplicate pfc_ids.")
    _assert(precision_at_k(ranked, ["a", "c"], 2) == 0.5, "P@2 unexpected.")
    _assert(recall_at_k(ranked, ["a", "c"], 2) == 0.5, "R@2 unexpected.")
    _assert(mean_reciprocal_rank(ranked, ["c"]) == 1.0 / 3.0, "MRR unexpected.")
    unsupported = score_ranking(ranked, [], unsupported=True)
    _assert(unsupported["mrr"] is None, "Unsupported MRR must be null.")
    _assert(unsupported["recall_at_5"] is None, "Unsupported Recall must be null.")


def main() -> int:
    test_retrieve_signature_default_is_hybrid()
    test_rag_service_calls_retrieve_without_mode_kwarg()
    test_rewrite_none_disables_rules_and_llm()
    test_rewrite_rules_still_expand_when_enabled()
    test_factory_defaults_keep_rewrite_enabled()
    test_dense_and_bm25_modes_skip_unused_channel()
    test_metrics_helpers()
    # Keep import referenced for clarity in failure messages.
    _assert(RAGService is not None, "RAGService import failed.")
    print("Phase 1 benchmark non-regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
