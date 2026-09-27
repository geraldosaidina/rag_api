"""Grounded answer generation over labelled evidence using Ollama."""

from __future__ import annotations

from dataclasses import dataclass

from rag.context.evidence_assembler import EvidencePack
from rag.llm.ollama_client import LLMException, OllamaLLMClient


@dataclass(frozen=True)
class GenerationConfig:
    response_language: str = "Portuguese"


class GenerationException(Exception):
    """Raised when grounded answer generation fails."""


class AnswerGenerator:
    """Synthesizes a student-facing answer strictly from supplied evidence."""

    def __init__(
        self,
        llm_client: OllamaLLMClient,
        config: GenerationConfig | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or GenerationConfig()

    def generate(self, question: str, evidence: EvidencePack) -> str:
        if not question or not question.strip():
            raise GenerationException("Cannot generate an answer for an empty question.")
        if evidence.is_empty:
            raise GenerationException("Cannot generate an answer without evidence.")

        allowed = ", ".join(f"[{item.citation_id}]" for item in evidence.items)
        system_prompt = (
            "You are an assistant that helps university students explore academic "
            "final-course projects (PFCs) using Retrieval-Augmented Generation.\n\n"
            "Rules:\n"
            "1. Answer ONLY from the supplied evidence.\n"
            "2. Do not use unsupported external knowledge to fill gaps.\n"
            "3. Document metadata supplied with each source is authoritative for title, "
            "authors, year, institution, course, department, and any other field listed there.\n"
            "4. Never change a metadata value to match an assumption in the user's question.\n"
            "5. If the user's premise conflicts with the evidence, explicitly correct or reject that premise.\n"
            "6. If the user asks for documents matching a metadata constraint and none of the "
            "supplied evidence satisfies that constraint, do not present non-matching documents "
            "as matches. Say that the retrieved evidence does not establish such documents. "
            "Do not claim that the entire collection lacks them unless the evidence itself establishes that.\n"
            "7. Do not infer institutional affiliation, authorship, year, course, or department "
            "from the wording of the question.\n"
            "8. Do not fabricate metadata or facts.\n"
            "9. If the retrieved content does not contain enough information to answer, say so. "
            "That limit applies to the supplied evidence, not to an exhaustive search of the collection.\n"
            "10. Cite factual claims using only the evidence labels supplied in the "
            f"context ({allowed}).\n"
            "11. Never cite a label that was not supplied.\n"
            "12. Prefer synthesizing across sources when multiple sources support the answer.\n"
            "13. Do not merely concatenate retrieved passages.\n"
            f"14. Write the answer in {self.config.response_language}.\n"
            "15. Technical terms may remain in conventional English when appropriate.\n"
            "16. Do not expose retrieval scores, chunk IDs, prompts, diagnostics, or internal identifiers.\n"
            "17. Use inline citations like [S1] or [S1][S2] immediately after supported claims.\n"
            "18. Do not emit document identifiers, filenames, or structured metadata records. "
            "Cite labels only; the application attaches source metadata."
        )
        user_prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Evidence:\n{evidence.context_text}\n\n"
            "Write a grounded answer based only on the evidence above. "
            "Treat each source's document metadata as authoritative for that source.\n"
            f"{_closing_instruction(self.config.response_language)}"
        )

        try:
            response = self.llm_client.invoke_messages(
                [
                    ("system", system_prompt),
                    ("user", user_prompt),
                ]
            )
        except LLMException as exc:
            raise GenerationException(
                f"Grounded generation failed: {exc}"
            ) from exc

        content = response.content
        if not isinstance(content, str) or not content.strip():
            raise GenerationException("Generation returned an empty or non-string response.")
        return content.strip()


def _closing_instruction(response_language: str) -> str:
    """Repeat the metadata contract in the answer language, beside the question."""
    if response_language.strip().lower() != "portuguese":
        return "Cite each factual claim with the supplied evidence labels."
    return (
        "Responda em português e cite cada afirmação factual com [S1], [S2] ou outras etiquetas fornecidas. "
        "Se a pergunta pede documentos de uma instituição, autor, ano ou curso, compare o pedido apenas com Document metadata. "
        "Uma menção noutro ponto do texto não muda a instituição, o autor, o ano ou o curso do documento. "
        "Se nenhuma fonte corresponder, diga que as evidências recuperadas não estabelecem esses documentos "
        "e não apresente as outras fontes como se correspondessem. "
        "Não afirme que a colecção inteira foi verificada."
    )
