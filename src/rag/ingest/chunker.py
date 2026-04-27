import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Configuration for splitting extracted PDF text into smaller chunks.
    """

    chunk_size: int = 400
    chunk_overlap: int = 100


class ChunkingException(Exception):
    """Raised when document chunking fails."""


class DocumentChunker:
    """
    Responsible for converting page-level Documents into smaller chunk-level Documents.

    This class does NOT:
    - load PDFs
    - create embeddings
    - store documents in Chroma
    - call the LLM
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split many page-level Documents into chunk-level Documents.
        """

        if not documents:
            raise ChunkingException("Cannot chunk an empty document list.")

        try:
            logger.info(
                "Chunking %s documents with chunk_size=%s and chunk_overlap=%s.",
                len(documents),
                self.config.chunk_size,
                self.config.chunk_overlap,
            )

            chunked_documents: list[Document] = []

            for document in documents:
                chunks = self._split_document(document)
                chunked_documents.extend(chunks)

            logger.info("Created %s total chunks.", len(chunked_documents))

            return chunked_documents

        except ChunkingException:
            raise
        except Exception as exc:
            raise ChunkingException("Failed to chunk documents.") from exc

    def _split_document(self, document: Document) -> list[Document]:
        """
        Split one page-level Document into multiple chunk-level Documents.
        """

        if not document.page_content or not document.page_content.strip():
            return []

        split_texts = self._splitter.split_text(document.page_content)

        chunks: list[Document] = []

        for chunk_index, text in enumerate(split_texts):
            cleaned_text = self._clean_text(text)

            if not cleaned_text:
                continue

            metadata = dict(document.metadata)
            metadata["source"] = metadata.get("source", "unknown-source")
            metadata["page"] = metadata.get("page", "unknown-page")
            metadata["chunk_index"] = chunk_index
            metadata["section_type"] = self._classify_section_type(cleaned_text, metadata)
            metadata["content_type"] = self._classify_content_type(cleaned_text, metadata)
            metadata["retrieval_quality"] = self._classify_retrieval_quality(
                cleaned_text,
                metadata["section_type"],
                metadata["content_type"],
            )
            metadata["chunk_id"] = self._build_chunk_id(
                content=cleaned_text,
                metadata=metadata,
            )

            chunks.append(
                Document(
                    page_content=cleaned_text,
                    metadata=metadata,
                )
            )

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Remove common PDF extraction artifacts while keeping readable content.
        """

        cleaned = text

        known_noise_patterns = [
            r"conference acronym[^.!\n]{0,120}",
            r"\bPriprint\b",
            r"[‘']?xx,\s*june\s*\d{2}\s*[–-]\s*\d{2},\s*\d{4},\s*woodstock,\s*ny",
        ]
        for pattern in known_noise_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"(?:[✓✗]\s*){5,}", " ", cleaned)
        cleaned = re.sub(
            r"^\s*A Survey on Knowledge-Oriented Retrieval-Augmented Generation\b[:\s\-–—]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        leading_window = cleaned[:280]
        table_symbol_count = len(re.findall(r"[✓✗]", leading_window))
        citation_count = len(re.findall(r"\[\d{1,4}\]", leading_window))
        compact_token_count = len(re.findall(r"\b(?:\d{4}|[A-Z][a-z]{1,8}|✓|✗)\b", leading_window))
        if (
            table_symbol_count >= 4
            or citation_count >= 3
            or ("table" in leading_window.lower() and compact_token_count >= 14)
        ):
            sentence_split = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
            if len(sentence_split) == 2 and len(sentence_split[1].strip()) >= 80:
                cleaned = sentence_split[1]

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        deduped_lines: list[str] = []
        for line in lines:
            lower = line.lower()
            if deduped_lines and lower == deduped_lines[-1].lower():
                continue
            deduped_lines.append(line)

        cleaned = " ".join(deduped_lines)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"^[\s\-\|:;,.]+", "", cleaned)
        cleaned = re.sub(r"[\s\-\|:;,.]+$", "", cleaned)

        return cleaned.strip()

    def _classify_content_type(self, text: str, metadata: dict[str, Any]) -> str:
        lower_text = text.lower()
        prefix = lower_text[:300]
        page_indicator = str(metadata.get("page", "")).lower()

        boilerplate_terms = [
            "copyright",
            "permission to make digital or hard copies",
            "all rights reserved",
            "acm",
            "authors' addresses",
        ]
        author_block_like = bool(
            re.search(r"\b[a-z]+,\s+[a-z]+,\s+[a-z]+,?\s+and\s+[a-z]+\b", prefix)
        ) and "university" in lower_text
        if any(term in lower_text for term in boilerplate_terms) or author_block_like:
            return "boilerplate"

        checkmark_count = len(re.findall(r"[✓✗]", text))
        compact_rows = len(
            re.findall(
                r"(?i)(?:\b(?:\d{4}|yes|no|ours|baseline|et al\.?|✓|✗)\b[\s,;|/]*){5,}",
                text,
            )
        )
        has_table_heading = "table" in prefix or "table" in page_indicator
        looks_like_table = (
            checkmark_count >= 6
            or compact_rows >= 1
            or (has_table_heading and checkmark_count >= 3)
        )

        prose_tail = lower_text[220:]
        has_prose_tail = bool(
            re.search(
                r"\b(is|are|can|provides|improves|addresses|retrieval|generation)\b",
                prose_tail,
            )
        ) and len(prose_tail.split()) >= 25
        starts_table_like = (
            checkmark_count >= 3
            or len(re.findall(r"\[\d{1,4}\]", prefix)) >= 3
            or (has_table_heading and len(prefix.split()) <= 70)
        )
        if starts_table_like and has_prose_tail:
            return "mixed"
        if looks_like_table:
            return "table"
        return "normal"

    def _classify_section_type(self, text: str, metadata: dict[str, Any]) -> str:
        """
        Classify a chunk as body, references, or appendix using simple heuristics.
        """

        lower_text = text.lower()
        page = metadata.get("page")
        page_indicator = str(page).lower() if page is not None else ""

        heading_like_references = (
            "references" in lower_text[:200]
            or "bibliography" in lower_text[:200]
            or "works cited" in lower_text[:200]
            or "references" in page_indicator
            or "bibliography" in page_indicator
        )
        heading_like_appendix = (
            "appendix" in lower_text[:200]
            or "supplementary material" in lower_text[:250]
            or "supplementary" in lower_text[:150]
        )

        citation_markers = len(re.findall(r"\[\d{1,4}\]", text))
        numbered_reference_lines = len(
            re.findall(r"(?m)^\s*\[\d{1,4}\]\s", text)
        ) + len(re.findall(r"(?m)^\s*\d+\.\s+[A-Z]", text))
        link_markers = len(re.findall(r"(arxiv|doi|https?://)", lower_text))

        looks_like_references = (
            heading_like_references
            or numbered_reference_lines >= 2
            or (citation_markers >= 5 and link_markers >= 1)
        )

        if looks_like_references:
            return "references"
        if heading_like_appendix:
            return "appendix"
        return "body"

    def _classify_retrieval_quality(
        self, text: str, section_type: str, content_type: str
    ) -> str:
        """
        Mark low-value chunks (e.g. citation-heavy reference blocks) as low.
        """

        if section_type == "references":
            return "low"
        if content_type in {"table", "boilerplate"}:
            return "low"
        if content_type == "mixed":
            return "normal"

        lower_text = text.lower()
        citation_markers = len(re.findall(r"\[\d{1,4}\]", text))
        link_markers = len(re.findall(r"(arxiv|doi|https?://|www\.)", lower_text))
        numbered_reference_lines = len(
            re.findall(r"(?m)^\s*\[\d{1,4}\]\s", text)
        ) + len(re.findall(r"(?m)^\s*\d+\.\s+[A-Z]", text))
        total_length = max(len(text), 1)

        citation_ratio = citation_markers / total_length
        is_citation_heavy = (
            citation_markers >= 8
            or numbered_reference_lines >= 2
            or link_markers >= 3
            or citation_ratio >= 0.02
        )

        return "low" if is_citation_heavy else "normal"

    def build_ids(self, documents: list[Document]) -> list[str]:
        """
        Build stable IDs for Chroma from chunk metadata.

        Chroma can use these IDs to avoid random autogenerated IDs.
        """

        ids: list[str] = []

        for document in documents:
            chunk_id = document.metadata.get("chunk_id")

            if not chunk_id:
                raise ChunkingException(
                    "Cannot build Chroma IDs because a document is missing chunk_id."
                )

            ids.append(str(chunk_id))

        return ids

    def _build_chunk_id(self, content: str, metadata: dict[str, Any]) -> str:
        """
        Build a stable chunk ID from source, page, chunk index, and content hash.
        """

        source = metadata.get("source", "unknown-source")
        page = metadata.get("page", "unknown-page")
        chunk_index = metadata.get("chunk_index", "unknown-chunk")

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

        return f"{source}:page-{page}:chunk-{chunk_index}:{content_hash}"