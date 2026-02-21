"""Earnings release ingestor extracting 8-K Exhibit 99 filings."""

import logging
from pathlib import Path

from llama_index.core.schema import Document
from sec_edgar_downloader import Downloader

from company_rag.config.settings import get_settings
from company_rag.ingestion.base import BaseIngestor

logger = logging.getLogger(__name__)


class EarningsReleaseIngestor(BaseIngestor):
    """Ingest earnings releases from SEC 8-K Exhibit 99 filings.

    Earnings releases are typically filed as exhibits (EX-99.1) attached to
    8-K filings. This ingestor downloads 8-K filings and extracts the
    exhibit content containing earnings announcements.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._user_agent = settings.sec_edgar_user_agent
        self._download_dir = Path(settings.ingestion_download_dir)
        self._years_back = settings.ingestion_years_back

    def ingest(self, ticker: str, **kwargs: object) -> list[Document]:
        """Download and parse 8-K Exhibit 99 earnings releases.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Optional overrides:
                - years_back: int — Number of years of filings to retrieve.

        Returns:
            List of Document objects with earnings release metadata.
        """
        years_back = kwargs.get("years_back", self._years_back)

        agent_parts = str(self._user_agent).split()
        company_name = agent_parts[0] if agent_parts else "CompanyRAG"
        email = agent_parts[1] if len(agent_parts) > 1 else "research@example.com"

        dl = Downloader(company_name=company_name, email_address=email)

        # 8-K filings can be frequent; estimate ~4 per quarter
        limit = int(years_back) * 16
        logger.info("Downloading 8-K filings for %s (limit=%d)", ticker, limit)
        dl.get("8-K", ticker, limit=limit, download_details=True)

        documents: list[Document] = []
        filing_dir = self._download_dir / "sec-edgar-filings" / ticker / "8-K"
        if not filing_dir.exists():
            logger.warning("No 8-K filings found at %s", filing_dir)
            return documents

        for exhibit_path in sorted(filing_dir.rglob("*")):
            if not exhibit_path.is_file():
                continue
            # Look for Exhibit 99 files (earnings releases)
            name_lower = exhibit_path.name.lower()
            if "ex99" in name_lower or "ex-99" in name_lower or "exhibit99" in name_lower:
                doc = self._parse_exhibit(exhibit_path, ticker)
                if doc:
                    documents.append(doc)

        logger.info("Ingested %d earnings release documents for %s", len(documents), ticker)
        return documents

    def _parse_exhibit(self, path: Path, ticker: str) -> Document | None:
        """Parse a single Exhibit 99 file into a Document.

        Args:
            path: Path to the exhibit file.
            ticker: Stock ticker symbol.

        Returns:
            Document with metadata, or None if parsing fails.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.exception("Failed to read exhibit at %s", path)
            return None

        if not text.strip():
            logger.warning("Empty exhibit at %s", path)
            return None

        accession_number = path.parent.name

        metadata = {
            "ticker": ticker,
            "source_type": "earnings_release",
            "filing_date": "",
            "section": "Exhibit 99",
            "source_url": "",
            "document_title": f"{ticker} Earnings Release — {accession_number}",
            "accession_number": accession_number,
            "file_path": str(path),
        }

        return Document(text=text, metadata=metadata)
