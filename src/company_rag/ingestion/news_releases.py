"""Company newsroom scraper for press/news releases."""

import hashlib
import logging

import requests
from bs4 import BeautifulSoup
from llama_index.core.schema import Document

from company_rag.config.settings import get_settings
from company_rag.ingestion.base import BaseIngestor

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30


class NewsReleaseIngestor(BaseIngestor):
    """Ingest news/press releases from a company's newsroom.

    Scrapes a curated list of newsroom URLs and extracts article text content.
    URLs are configured via settings (ingestion_website_urls) or passed directly.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._urls: list[str] = settings.ingestion_website_urls
        self._user_agent = settings.sec_edgar_user_agent

    def ingest(self, ticker: str, **kwargs: object) -> list[Document]:
        """Scrape news release pages and create Documents.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Optional overrides:
                - urls: list[str] — List of newsroom URLs to scrape.

        Returns:
            List of Document objects with news release metadata.
        """
        urls = kwargs.get("urls", self._urls)
        if not urls:
            logger.warning("No newsroom URLs configured for %s", ticker)
            return []

        documents: list[Document] = []
        for url in urls:
            doc = self._scrape_page(str(url), ticker)
            if doc:
                documents.append(doc)

        logger.info("Ingested %d news release documents for %s", len(documents), ticker)
        return documents

    def _scrape_page(self, url: str, ticker: str) -> Document | None:
        """Scrape a single news release page.

        Args:
            url: URL of the news release page.
            ticker: Stock ticker symbol.

        Returns:
            Document with metadata, or None if scraping fails.
        """
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self._user_agent},
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Failed to fetch %s", url)
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Extract main text content
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if not text:
            logger.warning("No text content extracted from %s", url)
            return None

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

        metadata = {
            "ticker": ticker,
            "source_type": "news_release",
            "filing_date": "",
            "section": "",
            "source_url": url,
            "document_title": title,
            "url_hash": url_hash,
        }

        return Document(text=text, metadata=metadata)
