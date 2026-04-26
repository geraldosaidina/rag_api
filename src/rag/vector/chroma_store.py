import logging
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromaConfig:
    """
    Configuration for the local Chroma vector database.

    The vector store is persisted on disk so that you do not need to re-ingest
    all PDFs every time the application restarts.
    """

    persist_directory: str = "data/chroma"
    collection_name: str = "literature_review"

@dataclass(frozen=True)
class SearchResult:
    """
    Normalized result returned by the vector store.

    Upper layers should consume this instead of depending directly on Chroma or
    LangChain's raw return format.
    """

    content: str
    metadata: dict[str, Any]
    score: float | None = None


class ChromaException(Exception):
    """Raised when the Chroma vector database cannot complete an operation."""


def build_chroma_client(
    config: ChromaConfig,
    embedding_function: Embeddings,
) -> Chroma:
    """
    Factory function for creating the raw LangChain Chroma client.

    This is the only place where the Chroma object should be instantiated.
    """

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
        self._client = build_chroma_client(
            config=self.config,
            embedding_function=self.embedding_function,
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Add documents/chunks to the vector store.

        During ingestion, each PDF chunk becomes a LangChain Document containing:
        - page_content
        - metadata
        """

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
        """
        Search for documents semantically similar to the query.

        This returns documents without scores. It is useful when you only care
        about the content and metadata.
        """

        if not query or not query.strip():
            raise ChromaException("Cannot search with an empty query.")

        if k <= 0:
            raise ChromaException("Search parameter 'k' must be greater than zero.")

        try:
            logger.info(
                "Running similarity search in Chroma collection '%s' with k=%s.",
                self.config.collection_name,
                k,
            )

            documents = self._client.similarity_search(
                query=query,
                k=k,
                filter=metadata_filter,
            )

            return [
                SearchResult(
                    content=document.page_content,
                    metadata=document.metadata,
                    score=None,
                )
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
        """
        Search for documents semantically similar to the query and include scores.

        Use this during development because scores help you debug retrieval
        quality.
        """

        if not query or not query.strip():
            raise ChromaException("Cannot search with an empty query.")

        if k <= 0:
            raise ChromaException("Search parameter 'k' must be greater than zero.")

        try:
            logger.info(
                "Running scored similarity search in Chroma collection '%s' with k=%s.",
                self.config.collection_name,
                k,
            )

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

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None):
        """
        Return a LangChain retriever.

        This is useful later if you want to compose this vector store directly
        with LangChain chains or retrieval pipelines.
        """

        return self._client.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )

    def count(self) -> int:
        """
        Return the number of stored items in the Chroma collection.

        Useful for health checks and debugging ingestion.
        """

        try:
            collection = self._client._collection
            return collection.count()
        except Exception as exc:
            raise ChromaException(
                f"Failed to count documents in Chroma collection '{self.config.collection_name}'."
            ) from exc

    def health_check(self) -> bool:
        """
        Verify that the Chroma collection is reachable.

        This does not prove retrieval quality. It only proves that the vector
        store can be accessed.
        """

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