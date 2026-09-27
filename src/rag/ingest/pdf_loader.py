import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFLoaderException(Exception):
    """Raised when PDF loading or parsing fails."""


class PDFLoader:
    """
    Responsible for loading PDF files and converting them into LangChain Document objects.

    This class does NOT:
    - split text into chunks
    - embed text
    - store anything in a vector DB

    It only extracts raw text + metadata from PDFs.
    """

    def __init__(self, data_dir: str = "data/raw_pdfs"):
        """
        Initialize the loader with a directory containing PDF files.

        Args:
            data_dir: Path to the folder where PDFs are stored.
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise PDFLoaderException(
                f"PDF directory does not exist: {self.data_dir}"
            )

    def list_pdf_files(self) -> List[Path]:
        """
        List all PDF files in the data directory.

        Returns:
            List of file paths for all PDFs.
        """
        return list(self.data_dir.glob("*.pdf"))

    def load_pdf(self, file_path: Path, *, skip_page_errors: bool = False) -> List[Document]:
        """
        Load a single PDF file and extract its content page by page.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page.
        """
        try:
            logger.info("Loading PDF: %s", file_path.name)
            if skip_page_errors:
                return self._load_pages_lenient(file_path)

            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            for i, doc in enumerate(documents):
                doc.metadata["source"] = file_path.name
                doc.metadata["page"] = i + 1  # physical PDF page index

            return documents

        except Exception as exc:
            raise PDFLoaderException(
                f"Failed to load PDF: {file_path.name}"
            ) from exc

    def _load_pages_lenient(self, file_path: Path) -> List[Document]:
        """Keep unreadable pages empty so one bad page does not abort ingestion."""
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        documents: List[Document] = []
        failed_pages = 0
        failure_type = "Exception"
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                failed_pages += 1
                failure_type = type(exc).__name__
                text = ""
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name, "page": index + 1},
                )
            )
        if failed_pages:
            logger.warning(
                "Skipped %s unreadable page(s) of %s: %s",
                failed_pages,
                file_path.name,
                failure_type,
            )
        return documents

    def load_all_pdfs(self) -> List[Document]:
        """
        Load all PDFs in the directory.

        Returns:
            A combined list of Document objects from all PDFs.
        """
        all_documents: List[Document] = []

        pdf_files = self.list_pdf_files()

        if not pdf_files:
            raise PDFLoaderException(
                f"No PDF files found in directory: {self.data_dir}"
            )

        for file_path in pdf_files:
            documents = self.load_pdf(file_path)
            all_documents.extend(documents)

        logger.info("Loaded %s total pages from PDFs", len(all_documents))

        return all_documents