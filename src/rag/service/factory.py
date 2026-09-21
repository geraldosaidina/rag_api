"""One-shot construction of a reusable RAG stack for CLI and future FastAPI."""

from __future__ import annotations

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
    response_language: str = "Portuguese"
    candidate_k: int = 40
    rerank_top_k: int = 8
    max_evidence_chunks: int = 5
    max_evidence_chars: int = 6000
    use_llm_query_rewrite: bool = True


def build_rag_service(config: RAGAppConfig | None = None) -> RAGService:
    """
    Build the full RAG QA stack once.

    Heavyweight components (embeddings, Chroma, reranker, Ollama) are created here
    and reused across subsequent answer() calls.
    """
    cfg = config or RAGAppConfig()

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

    llm_client = OllamaLLMClient(LLMConfig(model_name=cfg.llm_model_name))
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
