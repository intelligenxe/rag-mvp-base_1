"""Abstract base class for data ingestion."""

from abc import ABC, abstractmethod

from llama_index.core.schema import Document


class BaseIngestor(ABC):
    """Abstract interface for document ingestion.

    Implementations:
        - SECFilingIngestor (sec_filings.py): SEC EDGAR filing downloader
        - EarningsReleaseIngestor (earnings_releases.py): 8-K Exhibit 99 extraction
        - NewsReleaseIngestor (news_releases.py): Company newsroom scraper
        - WebsiteIngestor (website.py): Curated company page scraper

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseIngestor
        3. Implement all abstract methods
    """

    @abstractmethod
    def ingest(self, ticker: str, **kwargs: object) -> list[Document]:
        """Ingest documents for a given company ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Additional parameters specific to the ingestor.

        Returns:
            List of LlamaIndex Document objects with metadata.
        """
        ...
