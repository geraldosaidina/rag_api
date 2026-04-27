import sys
from pathlib import Path

# Ensure imports work when running from repository root:
# uv run python src/scripts/test_retrieval.py
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.embeddings.embedder import EmbeddingConfig, build_ollama_embeddings_client
from rag.vector.chroma_store import ChromaConfig, ChromaVectorStore


def main() -> int:
    # IMPORTANT:
    # If your collection was ingested before section_type/retrieval_quality
    # metadata existed, clear/rebuild data/chroma (or use another persist
    # directory) before validating filtered retrieval behavior.
    #
    # Validation runbook:
    # 1) Run: uv run python src/scripts/ingest.py
    # 2) Run this script
    # 3) Compare RAW RESULTS vs FILTERED RESULTS for the same query
    # 4) Confirm FILTERED RESULTS remove references/low-quality chunks
    #    and prioritize body chunks for answer-bearing passages.
    embedding_fn = build_ollama_embeddings_client(
        EmbeddingConfig(model_name="BAAI/bge-m3")
    )

    store = ChromaVectorStore(
        embedding_function=embedding_fn,
        config=ChromaConfig(
            persist_directory="data/chroma",
            collection_name="literature_review",
        ),
    )

    print(f"Stored chunks: {store.count()}")

    queries = [
        "How does the author define Retrieval-Augmented Generation?",
        "What are hallucinations?",
    ]

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("RAW RESULTS")
        print("-" * 100)

        raw_results = store.similarity_search_with_scores(
            query=query,
            k=4,
            metadata_filter={"source": "2503.10677v2.pdf"},
        )

        for index, result in enumerate(raw_results, 1):
            preview = result.content.replace("\n", " ").strip()
            if len(preview) > 280:
                preview = preview[:280] + "..."

            print(
                f"{index}. score={result.score:.6f} | "
                f"adjusted_score={_format_optional_score(result.adjusted_score)} | "
                f"page={result.metadata.get('page')} | "
                f"chunk_index={result.metadata.get('chunk_index')} | "
                f"section_type={result.metadata.get('section_type', 'body')} | "
                f"retrieval_quality={result.metadata.get('retrieval_quality', 'normal')} | "
                f"chunk_id={result.metadata.get('chunk_id')}"
            )
            print(f"   {preview}\n")

        print("FILTERED RESULTS")
        print("-" * 100)
        filtered_results = store.similarity_search_filtered(
            query=query,
            final_k=4,
            candidate_k=12,
            metadata_filter={"source": "2503.10677v2.pdf"},
        )

        for index, result in enumerate(filtered_results, 1):
            preview = result.content.replace("\n", " ").strip()
            if len(preview) > 280:
                preview = preview[:280] + "..."

            print(
                f"{index}. score={result.score:.6f} | "
                f"adjusted_score={_format_optional_score(result.adjusted_score)} | "
                f"page={result.metadata.get('page')} | "
                f"chunk_index={result.metadata.get('chunk_index')} | "
                f"section_type={result.metadata.get('section_type', 'body')} | "
                f"retrieval_quality={result.metadata.get('retrieval_quality', 'normal')} | "
                f"chunk_id={result.metadata.get('chunk_id')}"
            )
            print(f"   {preview}\n")

    return 0


def _format_optional_score(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
