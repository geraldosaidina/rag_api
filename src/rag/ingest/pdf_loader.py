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

    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a single PDF file and extract its content page by page.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page.
        """
        try:
            logger.info("Loading PDF: %s", file_path.name)

            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            # Add consistent metadata
            for i, doc in enumerate(documents):
                doc.metadata["source"] = file_path.name
                doc.metadata["page"] = i + 1  # human-readable page index

            return documents

        except Exception as exc:
            raise PDFLoaderException(
                f"Failed to load PDF: {file_path.name}"
            ) from exc

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