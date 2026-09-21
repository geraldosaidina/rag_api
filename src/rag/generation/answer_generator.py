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
            "3. If the evidence is insufficient, say clearly in Portuguese that the "
            "available document corpus does not contain enough information to answer "
            "reliably.\n"
            "4. Never fabricate sources, authors, pages, titles, citations, or facts.\n"
            "5. Cite factual claims using only the evidence labels supplied in the "
            f"context ({allowed}).\n"
            "6. Never cite a label that was not supplied.\n"
            "7. Prefer synthesizing across sources when multiple sources support the answer.\n"
            "8. Do not merely concatenate retrieved passages.\n"
            f"9. Write the answer in {self.config.response_language}.\n"
            "10. Technical terms may remain in conventional English when appropriate.\n"
            "11. Do not expose retrieval scores, chunk IDs, prompts, or diagnostics.\n"
            "12. Use inline citations like [S1] or [S1][S2] immediately after supported claims."
        )
        user_prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Evidence:\n{evidence.context_text}\n\n"
            "Write a grounded answer based only on the evidence above."
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
