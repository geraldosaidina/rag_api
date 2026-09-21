"""One-shot construction of a reusable RAG stack for CLI and FastAPI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from rag.citations.citation_validator import CitationValidator
from rag.context.evidence_assembler import EvidenceAssembler, EvidenceAssemblerConfig
from rag.embeddings.embedder import EmbeddingConfig, build_embeddings_client
from rag.generation.answer_generator import AnswerGenerator, GenerationConfig
from rag.llm.ollama_client import LLMConfig, OllamaLLMClient
from rag.query.query_rewriter import QueryRewriteConfig, QueryRewriter
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.rerank.reranker import CrossEncoderReranker, RerankerConfig
from rag.service.rag_service import RAGService, RAGServiceConfig
from rag.vector.chroma_store import ChromaConfig, ChromaVectorStore


@dataclass(frozen=True)
class RAGAppConfig:
    persist_directory: str = "data/chroma"
    collection_name: str = "pfc_corpus"
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    llm_model_name: str = "llama3.1:8b"
    ollama_base_url: str | None = None
    response_language: str = "Portuguese"
    candidate_k: int = 40
    rerank_top_k: int = 8
    max_evidence_chunks: int = 5
    max_evidence_chars: int = 6000
    use_llm_query_rewrite: bool = True


def load_rag_app_config() -> RAGAppConfig:
    """Load RAG runtime config from environment with safe local defaults."""
    return RAGAppConfig(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pfc_corpus"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        reranker_model_name=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        llm_model_name=os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL") or None,
        response_language=os.getenv("RESPONSE_LANGUAGE", "Portuguese"),
        candidate_k=int(os.getenv("RAG_CANDIDATE_K", "40")),
        rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "8")),
        max_evidence_chunks=int(os.getenv("RAG_MAX_EVIDENCE_CHUNKS", "5")),
        max_evidence_chars=int(os.getenv("RAG_MAX_EVIDENCE_CHARS", "6000")),
        use_llm_query_rewrite=os.getenv("RAG_USE_LLM_REWRITE", "true").lower()
        in {"1", "true", "yes"},
    )


def build_rag_service(config: RAGAppConfig | None = None) -> RAGService:
    """
    Build the full RAG QA stack once.

    Heavyweight components (embeddings, Chroma, reranker, Ollama) are created here
    and reused across subsequent answer() calls.
    """
    cfg = config or load_rag_app_config()

    embedding_fn = build_embeddings_client(
        EmbeddingConfig(model_name=cfg.embedding_model_name)
    )
    vector_store = ChromaVectorStore(
        embedding_function=embedding_fn,
        config=ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        ),
    )

    llm_client = OllamaLLMClient(
        LLMConfig(model_name=cfg.llm_model_name, base_url=cfg.ollama_base_url)
    )
    query_rewriter = QueryRewriter(
        llm_client=llm_client if cfg.use_llm_query_rewrite else None,
        config=QueryRewriteConfig(use_llm=cfg.use_llm_query_rewrite),
    )
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_function=embedding_fn,
        query_rewriter=query_rewriter,
    )
    reranker = CrossEncoderReranker(
        RerankerConfig(model_name=cfg.reranker_model_name)
    )
    evidence_assembler = EvidenceAssembler(
        EvidenceAssemblerConfig(
            max_chunks=cfg.max_evidence_chunks,
            max_chars=cfg.max_evidence_chars,
        )
    )
    answer_generator = AnswerGenerator(
        llm_client=llm_client,
        config=GenerationConfig(response_language=cfg.response_language),
    )

    return RAGService(
        retriever=retriever,
        reranker=reranker,
        evidence_assembler=evidence_assembler,
        answer_generator=answer_generator,
        citation_validator=CitationValidator(),
        config=RAGServiceConfig(
            candidate_k=cfg.candidate_k,
            rerank_top_k=cfg.rerank_top_k,
            max_evidence_chunks=cfg.max_evidence_chunks,
            max_evidence_chars=cfg.max_evidence_chars,
        ),
    )
