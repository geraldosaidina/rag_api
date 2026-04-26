import logging
from dataclasses import dataclass
from typing import Any

from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for the local Ollama embedding model.

    This is intentionally separate from the LLM config because the generation
    model and the embedding model have different responsibilities.
    """

    model_name: str = "nomic-embed-text"
    base_url: str | None = None


@dataclass(frozen=True)
class EmbeddingResponse:
    """
    Normalized response for a single embedding operation.

    This keeps upper layers from depending directly on LangChain internals.
    """

    embedding: list[float]
    model_name: str
    raw_response: Any | None = None


@dataclass(frozen=True)
class BatchEmbeddingResponse:
    """
    Normalized response for embedding multiple texts.

    This will be used during ingestion, when many document chunks need to be
    embedded before being stored in the vector database.
    """

    embeddings: list[list[float]]
    model_name: str
    raw_response: Any | None = None


class EmbeddingException(Exception):
    """Raised when the embedding client cannot complete an operation."""


def build_ollama_embeddings_client(config: EmbeddingConfig) -> OllamaEmbeddings:
    """
    Factory function for creating the raw LangChain OllamaEmbeddings client.

    This should be the only place where OllamaEmbeddings is instantiated.
    """

    try:
        return OllamaEmbeddings(
            model=config.model_name,
            base_url=config.base_url,
        )
    except Exception as exc:
        raise EmbeddingException(
            f"Failed to build Ollama embeddings client for model '{config.model_name}'."
        ) from exc


class OllamaEmbeddingClient:
    """
    App-facing embedding adapter.

    This class hides LangChain/Ollama details from the rest of the application.
    Ingestion and query layers should use this class instead of using
    OllamaEmbeddings directly.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._client = build_ollama_embeddings_client(self.config)

    def embed_query(self, text: str) -> EmbeddingResponse:
        """
        Embed one piece of text.

        This is mainly used at query time, when the user asks a question and
        the system needs to convert that question into a vector for retrieval.
        """

        if not text or not text.strip():
            raise EmbeddingException("Cannot embed an empty query.")

        try:
            logger.info("Embedding query with model: %s", self.config.model_name)

            embedding = self._client.embed_query(text)

            if not isinstance(embedding, list) or not all(
                isinstance(value, float) for value in embedding
            ):
                raise EmbeddingException(
                    f"Unexpected embedding format from model '{self.config.model_name}'."
                )

            return EmbeddingResponse(
                embedding=embedding,
                model_name=self.config.model_name,
                raw_response=embedding,
            )

        except EmbeddingException:
            raise
        except Exception as exc:
            raise EmbeddingException(
                f"Failed to embed query with model '{self.config.model_name}'."
            ) from exc

    def embed_documents(self, texts: list[str]) -> BatchEmbeddingResponse:
        """
        Embed multiple texts.

        This is mainly used during ingestion, when PDF chunks are converted into
        vectors before being stored in Chroma.
        """

        cleaned_texts = [text for text in texts if text and text.strip()]

        if not cleaned_texts:
            raise EmbeddingException("Cannot embed an empty document batch.")

        try:
            logger.info(
                "Embedding %s documents with model: %s",
                len(cleaned_texts),
                self.config.model_name,
            )

            embeddings = self._client.embed_documents(cleaned_texts)

            if not isinstance(embeddings, list) or not embeddings:
                raise EmbeddingException(
                    f"Unexpected batch embedding format from model '{self.config.model_name}'."
                )

            if not all(isinstance(vector, list) for vector in embeddings):
                raise EmbeddingException(
                    f"Expected a list of embedding vectors from model '{self.config.model_name}'."
                )

            return BatchEmbeddingResponse(
                embeddings=embeddings,
                model_name=self.config.model_name,
                raw_response=embeddings,
            )

        except EmbeddingException:
            raise
        except Exception as exc:
            raise EmbeddingException(
                f"Failed to embed documents with model '{self.config.model_name}'."
            ) from exc

    def health_check(self) -> bool:
        """
        Verifies that Ollama and the configured embedding model are reachable.

        This can later be used by a FastAPI health endpoint.
        """

        try:
            response = self.embed_query("health check")
            return bool(response.embedding)

        except Exception as exc:
            logger.warning(
                "Ollama embedding health check failed for model '%s': %s",
                self.config.model_name,
                exc,
            )
            return False