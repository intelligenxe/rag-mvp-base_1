"""SEC EDGAR filing ingestor using sec-edgar-downloader."""

import logging
from pathlib import Path

from llama_index.core.schema import Document
from sec_edgar_downloader import Downloader

from company_rag.config.settings import get_settings
from company_rag.ingestion.base import BaseIngestor

logger = logging.getLogger(__name__)


class SECFilingIngestor(BaseIngestor):
    """Ingest SEC filings (10-K, 10-Q, 8-K, etc.) from EDGAR.

    Downloads filings using sec-edgar-downloader and extracts text content
    from the downloaded files, creating LlamaIndex Document objects with
    appropriate metadata.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._user_agent = settings.sec_edgar_user_agent
        self._download_dir = Path(settings.ingestion_download_dir)
        self._filing_types = settings.ingestion_filing_types
        self._years_back = settings.ingestion_years_back

    def ingest(self, ticker: str, **kwargs: object) -> list[Document]:
        """Download and parse SEC filings for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Optional overrides:
                - filing_types: list[str] — Filing types to download.
                - years_back: int — Number of years of filings to retrieve.

        Returns:
            List of Document objects with SEC filing metadata.
        """
        filing_types = kwargs.get("filing_types", self._filing_types)
        years_back = kwargs.get("years_back", self._years_back)

        # Parse user agent into company name and email
        agent_parts = str(self._user_agent).split()
        company_name = agent_parts[0] if agent_parts else "CompanyRAG"
        email = agent_parts[1] if len(agent_parts) > 1 else "research@example.com"

        dl = Downloader(company_name=company_name, email_address=email)

        documents: list[Document] = []
        for filing_type in filing_types:
            limit = int(years_back) * 4 if filing_type in ("10-Q",) else int(years_back)
            logger.info("Downloading %s filings for %s (limit=%d)", filing_type, ticker, limit)
            dl.get(filing_type, ticker, limit=limit, download_details=True)

            # Parse downloaded files from the sec-edgar-downloader output directory
            filing_dir = self._download_dir / "sec-edgar-filings" / ticker / filing_type
            if not filing_dir.exists():
                logger.warning("No filings found at %s", filing_dir)
                continue

            for filing_path in sorted(filing_dir.rglob("*.txt")):
                doc = self._parse_filing(filing_path, ticker, filing_type)
                if doc:
                    documents.append(doc)

        logger.info("Ingested %d SEC filing documents for %s", len(documents), ticker)
        return documents

    def _parse_filing(self, path: Path, ticker: str, filing_type: str) -> Document | None:
        """Parse a single SEC filing file into a Document.

        Args:
            path: Path to the filing text file.
            ticker: Stock ticker symbol.
            filing_type: SEC filing type (e.g., "10-K").

        Returns:
            Document with metadata, or None if parsing fails.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.exception("Failed to read filing at %s", path)
            return None

        if not text.strip():
            logger.warning("Empty filing at %s", path)
            return None

        # Extract accession number from directory structure
        accession_number = path.parent.name

        metadata = {
            "ticker": ticker,
            "source_type": filing_type,
            "filing_date": "",
            "section": "",
            "source_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}",
            "document_title": f"{ticker} {filing_type} — {accession_number}",
            "accession_number": accession_number,
            "file_path": str(path),
        }

        return Document(text=text, metadata=metadata)
