"""Company website scraper for curated pages."""

import hashlib
import logging

import requests
from bs4 import BeautifulSoup
from llama_index.core.schema import Document

from company_rag.config.settings import get_settings
from company_rag.ingestion.base import BaseIngestor

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30


class WebsiteIngestor(BaseIngestor):
    """Ingest content from curated company website pages.

    Scrapes a predefined list of URLs from the company's official website.
    This is not a full crawl — only the explicitly listed pages are scraped.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._urls: list[str] = settings.ingestion_website_urls
        self._user_agent = settings.sec_edgar_user_agent

    def ingest(self, ticker: str, **kwargs: object) -> list[Document]:
        """Scrape curated website pages and create Documents.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Optional overrides:
                - urls: list[str] — List of page URLs to scrape.

        Returns:
            List of Document objects with website page metadata.
        """
        urls = kwargs.get("urls", self._urls)
        if not urls:
            logger.warning("No website URLs configured for %s", ticker)
            return []

        documents: list[Document] = []
        for url in urls:
            doc = self._scrape_page(str(url), ticker)
            if doc:
                documents.append(doc)

        logger.info("Ingested %d website page documents for %s", len(documents), ticker)
        return documents

    def _scrape_page(self, url: str, ticker: str) -> Document | None:
        """Scrape a single website page.

        Args:
            url: URL of the page to scrape.
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

        # Remove non-content elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if not text:
            logger.warning("No text content extracted from %s", url)
            return None

        content_hash = hashlib.sha256((url + text).encode()).hexdigest()[:16]

        metadata = {
            "ticker": ticker,
            "source_type": "website",
            "filing_date": "",
            "section": "",
            "source_url": url,
            "document_title": title,
            "content_hash": content_hash,
        }

        return Document(text=text, metadata=metadata)
