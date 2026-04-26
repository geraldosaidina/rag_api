import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from rag.embeddings.embedder import EmbeddingConfig, build_ollama_embeddings_client
from rag.ingest.chunker import ChunkingConfig, DocumentChunker
from rag.ingest.pdf_loader import PDFLoader
from rag.vector.chroma_store import ChromaConfig, ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionConfig:
    """
    Configuration for the full ingestion pipeline.

    This config does not replace the lower-level configs.
    It coordinates the main values needed to run ingestion end to end.
    """

    pdf_directory: str = "data/raw_pdfs"
    persist_directory: str = "data/chroma"
    collection_name: str = "literature_review"
    embedding_model_name: str = "nomic-embed-text"
    chunk_size: int = 1000
    chunk_overlap: int = 150


@dataclass(frozen=True)
class IngestionResult:
    """
    Summary of an ingestion run.

    This makes the pipeline return useful information instead of only printing logs.
    """

    loaded_pages: int
    created_chunks: int
    stored_chunks: int
    collection_name: str
    persist_directory: str


class IngestionException(Exception):
    """Raised when the ingestion pipeline cannot complete successfully."""


class IngestionPipeline:
    """
    Orchestrates the full ingestion process.

    This class does NOT:
    - parse individual PDF internals itself
    - split text itself
    - embed text directly
    - call the LLM
    - expose HTTP endpoints

    It coordinates the components responsible for those lower-level tasks.
    """

    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()

        self.pdf_loader = PDFLoader(data_dir=self.config.pdf_directory)

        self.chunker = DocumentChunker(
            config=ChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        )

        self.embedding_function = build_ollama_embeddings_client(
            EmbeddingConfig(
                model_name=self.config.embedding_model_name,
            )
        )

        self.vector_store = ChromaVectorStore(
            embedding_function=self.embedding_function,
            config=ChromaConfig(
                persist_directory=self.config.persist_directory,
                collection_name=self.config.collection_name,
            ),
        )

    def run(self) -> IngestionResult:
        """
        Run the complete ingestion pipeline.

        Flow:
        1. Load PDFs into page-level Documents
        2. Split page-level Documents into chunk-level Documents
        3. Build stable chunk IDs
        4. Store chunks in Chroma
        5. Return ingestion summary
        """

        try:
            logger.info("Starting ingestion pipeline.")

            page_documents = self._load_documents()

            chunk_documents = self._chunk_documents(page_documents)

            chunk_ids = self._build_chunk_ids(chunk_documents)

            stored_ids = self._store_chunks(
                documents=chunk_documents,
                ids=chunk_ids,
            )

            result = IngestionResult(
                loaded_pages=len(page_documents),
                created_chunks=len(chunk_documents),
                stored_chunks=len(stored_ids),
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
            )

            logger.info(
                "Ingestion completed. pages=%s chunks=%s stored=%s collection=%s",
                result.loaded_pages,
                result.created_chunks,
                result.stored_chunks,
                result.collection_name,
            )

            return result

        except Exception as exc:
            raise IngestionException("Ingestion pipeline failed.") from exc

    def _load_documents(self) -> list[Document]:
        """
        Load all PDFs from the configured PDF directory.
        """

        logger.info("Loading PDFs from directory: %s", self.config.pdf_directory)

        documents = self.pdf_loader.load_all_pdfs()

        if not documents:
            raise IngestionException("No documents were loaded from PDFs.")

        return documents

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split page-level Documents into chunk-level Documents.
        """

        logger.info("Chunking %s loaded PDF pages.", len(documents))

        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            raise IngestionException("No chunks were created from the loaded documents.")

        return chunks

    def _build_chunk_ids(self, documents: list[Document]) -> list[str]:
        """
        Build stable IDs for the chunk-level Documents.
        """

        logger.info("Building stable chunk IDs for %s chunks.", len(documents))

        return self.chunker.build_ids(documents)

    def _store_chunks(
        self,
        documents: list[Document],
        ids: list[str],
    ) -> list[str]:
        """
        Store chunk-level Documents in Chroma.
        """

        logger.info(
            "Storing %s chunks in Chroma collection '%s'.",
            len(documents),
            self.config.collection_name,
        )

        return self.vector_store.add_documents(
            documents=documents,
            ids=ids,
        )