from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

@dataclass(frozen=True)
class ChromaConfig:
    """Configuration for the Chroma vector database."""
    persist_directory: str = "data/chroma"
    collection_name: str = "literature_review"

@dataclass(frozen=True)
class SearchResults:
    """Results from a vector search."""
    content: str
    metadata: dict
    score: float

class ChromaException(Exception):
    """Raised when the Chroma vector database cannot complete an operation."""

def build_chroma_client(config: ChromaConfig) -> Chroma:
    """Factory function for creating the raw LangChain Chroma client."""
    try:
        return Chroma(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
        )
    except Exception as exc:
        raise ChromaException(
            f"Failed to build Chroma client for collection '{config.collection_name}'."
        ) from exc

@dataclass(frozen=True)
class ChromaVectorStore:
    """App-facing vector store adapter."""
    def __init__(self, config: ChromaConfig):
        self.config = config or ChromaConfig()
        self._client = build_chroma_client(self.config)

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        """Add documents to the vector store."""
        try:
            logger.info("Adding %s documents to Chroma collection '%s'", len(documents), self.config.collection_name)
            if ids and len(ids) != len(documents):
                raise ChromaException("Number of document IDs must match the number of documents.")
            if not documents:
                raise ChromaException("Cannot add an empty document list.")
            if ids:
                return self._client.add_documents(documents, ids)
            else:
                