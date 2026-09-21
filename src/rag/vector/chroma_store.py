import logging
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromaConfig:
    """Local Chroma persistence settings."""

    persist_directory: str = "data/chroma"
    collection_name: str = "pfc_corpus"


@dataclass(frozen=True)
class SearchResult:
    """Normalized vector-store hit consumed by retrieval/rerank layers."""

    content: str
    metadata: dict[str, Any]
    score: float | None = None
    adjusted_score: float | None = None


class ChromaException(Exception):
    """Raised when the Chroma vector database cannot complete an operation."""


def build_chroma_client(config: ChromaConfig, embedding_function: Embeddings) -> Chroma:
    try:
        return Chroma(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embedding_function=embedding_function,
        )
    except Exception as exc:
        raise ChromaException(
            f"Failed to build Chroma client for collection '{config.collection_name}'."
        ) from exc


class ChromaVectorStore:
    """App-facing vector store adapter."""

    def __init__(
        self,
        embedding_function: Embeddings,
        config: ChromaConfig | None = None,
    ):
        self.config = config or ChromaConfig()
        self.embedding_function = embedding_function
        self._client = build_chroma_client(self.config, self.embedding_function)

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not documents:
            raise ChromaException("Cannot add an empty document list.")
        if ids is not None and len(ids) != len(documents):
            raise ChromaException(
                "Number of document IDs must match the number of documents."
            )
        try:
            logger.info(
                "Adding %s documents to Chroma collection '%s'.",
                len(documents),
                self.config.collection_name,
            )
            if ids is not None:
                return self._client.add_documents(documents=documents, ids=ids)
            return self._client.add_documents(documents=documents)
        except ChromaException:
            raise
        except Exception as exc:
            raise ChromaException(
                f"Failed to add documents to Chroma collection '{self.config.collection_name}'."
            ) from exc

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ChromaException("Cannot search with an empty query.")
        if k <= 0:
            raise ChromaException("Search parameter 'k' must be greater than zero.")
        try:
            documents = self._client.similarity_search(
                query=query,
                k=k,
                filter=metadata_filter,
            )
            return [
                SearchResult(content=document.page_content, metadata=document.metadata)
                for document in documents
            ]
        except Exception as exc:
            raise ChromaException(
                f"Failed to run similarity search in Chroma collection '{self.config.collection_name}'."
            ) from exc

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ChromaException("Cannot search with an empty query.")
        if k <= 0:
            raise ChromaException("Search parameter 'k' must be greater than zero.")
        try:
            results = self._client.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter,
            )
            return [
                SearchResult(
                    content=document.page_content,
                    metadata=document.metadata,
                    score=score,
                )
                for document, score in results
            ]
        except Exception as exc:
            raise ChromaException(
                f"Failed to run scored similarity search in Chroma collection '{self.config.collection_name}'."
            ) from exc

    def similarity_search_filtered(
        self,
        query: str,
        final_k: int = 4,
        candidate_k: int = 12,
        metadata_filter: dict[str, Any] | None = None,
        exclude_section_types: set[str] | None = None,
        exclude_retrieval_qualities: set[str] | None = None,
    ) -> list[SearchResult]:
        """Retrieve candidates, then drop low-value chunks by metadata."""
        if not query or not query.strip():
            raise ChromaException("Cannot search with an empty query.")
        if final_k <= 0:
            raise ChromaException("Search parameter 'final_k' must be greater than zero.")
        if candidate_k <= 0:
            raise ChromaException(
                "Search parameter 'candidate_k' must be greater than zero."
            )
        if candidate_k < final_k:
            raise ChromaException(
                "Search parameter 'candidate_k' must be greater than or equal to 'final_k'."
            )

        blocked_section_types = exclude_section_types or {"references"}
        blocked_retrieval_qualities = exclude_retrieval_qualities or {"low"}

        try:
            candidates = self._client.similarity_search_with_score(
                query=query,
                k=candidate_k,
                filter=metadata_filter,
            )
            kept: list[SearchResult] = []
            for document, score in candidates:
                section_type = document.metadata.get("section_type", "body")
                retrieval_quality = document.metadata.get("retrieval_quality", "normal")
                if section_type in blocked_section_types:
                    continue
                if retrieval_quality in blocked_retrieval_qualities:
                    continue
                kept.append(
                    SearchResult(
                        content=document.page_content,
                        metadata=document.metadata,
                        score=score,
                    )
                )
            return kept[:final_k]
        except Exception as exc:
            raise ChromaException(
                f"Failed to run filtered similarity search in Chroma collection '{self.config.collection_name}'."
            ) from exc

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None):
        return self._client.as_retriever(search_kwargs=search_kwargs or {"k": 4})

    def count(self) -> int:
        try:
            return self._client._collection.count()
        except Exception as exc:
            raise ChromaException(
                f"Failed to count documents in Chroma collection '{self.config.collection_name}'."
            ) from exc

    def health_check(self) -> bool:
        try:
            self.count()
            return True
        except Exception as exc:
            logger.warning(
                "Chroma health check failed for collection '%s': %s",
                self.config.collection_name,
                exc,
            )
            return False
