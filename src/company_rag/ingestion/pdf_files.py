"""PDF file ingestor for arbitrary PDF documents."""

import hashlib
import logging
from pathlib import Path

from llama_index.core.schema import Document

from company_rag.config.settings import get_settings
from company_rag.ingestion.base import BaseIngestor

logger = logging.getLogger(__name__)


class PDFFileIngestor(BaseIngestor):
    """Ingest content from local PDF files.

    Reads PDF files from a configurable list of file paths, extracts text
    using LlamaIndex's PDFReader, and creates Document objects with standard
    metadata for downstream processing.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._pdf_paths: list[str] = settings.ingestion_pdf_paths

    def ingest(self, ticker: str, **kwargs: object) -> list[Document]:
        """Read PDF files and create Documents.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Optional overrides:
                - pdf_paths: list[str] — List of PDF file paths to ingest.

        Returns:
            List of Document objects with PDF metadata.
        """
        from llama_index.readers.file import PDFReader

        pdf_paths = kwargs.get("pdf_paths", self._pdf_paths)
        if not pdf_paths:
            logger.warning("No PDF paths configured for %s", ticker)
            return []

        reader = PDFReader()
        documents: list[Document] = []

        for pdf_path in pdf_paths:
            path = Path(str(pdf_path))
            if not path.exists():
                logger.warning("PDF file not found: %s", path)
                continue
            if path.suffix.lower() != ".pdf":
                logger.warning("Skipping non-PDF file: %s", path)
                continue

            try:
                pages = reader.load_data(file=path)
            except Exception:
                logger.exception("Failed to read PDF: %s", path)
                continue

            # Combine all pages into a single document
            text = "\n\n".join(page.get_content() for page in pages)
            if not text.strip():
                logger.warning("No text content extracted from %s", path)
                continue

            content_hash = hashlib.sha256((str(path.resolve()) + text).encode()).hexdigest()[:16]

            metadata = {
                "ticker": ticker,
                "source_type": "pdf",
                "filing_date": "",
                "section": "",
                "source_url": str(path.resolve()),
                "document_title": path.stem,
                "content_hash": content_hash,
            }

            documents.append(Document(text=text, metadata=metadata))

        logger.info("Ingested %d PDF documents for %s", len(documents), ticker)
        return documents
