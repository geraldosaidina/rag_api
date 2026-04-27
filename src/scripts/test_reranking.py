import sys
from pathlib import Path

# Ensure imports work when running from repository root:
# uv run python src/scripts/test_reranking.py
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.embeddings.embedder import EmbeddingConfig, build_ollama_embeddings_client
from rag.llm.ollama_client import OllamaLLMClient
from rag.query.query_rewriter import QueryRewriter
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.rerank.reranker import CrossEncoderReranker, RerankerConfig
from rag.vector.chroma_store import ChromaConfig, ChromaVectorStore, SearchResult


def _preview(text: str, max_chars: int = 280) -> str:
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "..."


def _print_candidate_results(title: str, results: list[SearchResult]) -> None:
    print(title)
    print("-" * 100)
    if not results:
        print("No results.\n")
        return

    for rank, result in enumerate(results, 1):
        print(
            f"{rank}. original_score={_format_optional_score(result.score)} | "
            f"dense_score={_format_optional_score(result.metadata.get('dense_score'))} | "
            f"bm25_score={_format_optional_score(result.metadata.get('bm25_score'))} | "
            f"answerability_boost={_format_optional_score(result.metadata.get('answerability_boost'))} | "
            f"combined_score={_format_optional_score(result.metadata.get('combined_score'))} | "
            f"retrieval_intent={result.metadata.get('retrieval_intent')} | "
            f"matched_query={result.metadata.get('matched_query')} | "
            f"content_type={result.metadata.get('content_type', 'normal')} | "
            f"source={result.metadata.get('source')} | "
            f"page={result.metadata.get('page')} | "
            f"chunk_index={result.metadata.get('chunk_index')} | "
            f"section_type={result.metadata.get('section_type', 'body')} | "
            f"retrieval_quality={result.metadata.get('retrieval_quality', 'normal')} | "
            f"chunk_id={result.metadata.get('chunk_id')}"
        )
        print(f"   {_preview(result.content)}\n")


def main() -> int:
    candidate_k = 100
    final_k = 5
    embedding_config = EmbeddingConfig(model_name="BAAI/bge-m3")
    reranker_config = RerankerConfig(model_name="BAAI/bge-reranker-v2-m3")

    embedding_fn = build_ollama_embeddings_client(embedding_config)
    vector_store = ChromaVectorStore(
        embedding_function=embedding_fn,
        config=ChromaConfig(
            persist_directory="data/chroma",
            collection_name="literature_review",
        ),
    )
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_function=embedding_fn,
        query_rewriter=QueryRewriter(llm_client=OllamaLLMClient()),
    )
    reranker = CrossEncoderReranker(reranker_config)

    print(
        "If you changed embedding_model_name or chunking settings "
        "(chunk_size/chunk_overlap), delete data/chroma and re-run ingestion "
        "before testing retrieval."
    )
    print(f"Stored chunks: {vector_store.count()}")
    print(
        f"candidate_k={candidate_k} final_k={final_k} "
        f"embedding_model={embedding_config.model_name} "
        f"reranker_model={reranker_config.model_name}"
    )

    queries = [
        "How does the author define Retrieval-Augmented Generation?",
        "What are hallucinations?",
        "What are the core components of RAG?",
        "How does RAG reduce hallucinations?",
    ]

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print(f"candidate_k={candidate_k} final_k={final_k}")

        filtered_candidates = hybrid_retriever.retrieve(
            query=query,
            candidate_k=candidate_k,
        )
        rewritten_queries = filtered_candidates[0].metadata.get("rewritten_queries", [query]) if filtered_candidates else [query]
        used_llm_rewrite = bool(filtered_candidates[0].metadata.get("used_llm_query_rewrite", False)) if filtered_candidates else False
        print(f"rewritten_queries={rewritten_queries}")
        print(f"used_llm_query_rewrite={used_llm_rewrite}")
        filtered_candidates = [
            result
            for result in filtered_candidates
            if result.metadata.get("source") == "2503.10677v2.pdf"
        ]

        _print_candidate_results(
            "FILTERED CANDIDATES (BEFORE RERANKING)",
            filtered_candidates,
        )

        reranked = reranker.rerank(
            query=query,
            candidates=filtered_candidates,
            top_k=final_k,
        )

        print("RERANKED RESULTS (FINAL)")
        print("-" * 100)
        if not reranked:
            print("No reranked results.\n")
            continue

        for rank, result in enumerate(reranked, 1):
            print(
                f"{rank}. original_score={_format_optional_score(result.original_score)} | "
                f"dense_score={_format_optional_score(result.metadata.get('dense_score'))} | "
                f"bm25_score={_format_optional_score(result.metadata.get('bm25_score'))} | "
                f"combined_score={_format_optional_score(result.metadata.get('combined_score'))} | "
                f"rerank_score={result.rerank_score:.6f} | "
                f"answerability_score={result.answerability_score:.6f} | "
                f"final_score={result.final_score:.6f} | "
                f"matched_query={result.metadata.get('matched_query')} | "
                f"content_type={result.metadata.get('content_type', 'normal')} | "
                f"source={result.metadata.get('source')} | "
                f"page={result.metadata.get('page')} | "
                f"chunk_index={result.metadata.get('chunk_index')} | "
                f"section_type={result.metadata.get('section_type', 'body')} | "
                f"retrieval_quality={result.metadata.get('retrieval_quality', 'normal')} | "
                f"chunk_id={result.metadata.get('chunk_id')}"
            )
            print(f"   {_preview(result.content)}\n")

    return 0


def _format_optional_score(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
