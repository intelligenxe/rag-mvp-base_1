"""Text cleaning utilities for preprocessing."""

import logging
import re

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

# Regex to strip HTML tags
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Regex to normalize whitespace (multiple spaces/tabs to single space)
_WHITESPACE_RE = re.compile(r"[ \t]+")

# Regex to normalize multiple newlines to at most two
_NEWLINE_RE = re.compile(r"\n{3,}")


class TextCleaner:
    """Clean raw text from ingested documents.

    Performs HTML tag stripping, whitespace normalization, and encoding
    issue handling. Applied to documents before chunking.
    """

    def clean(self, documents: list[Document]) -> list[Document]:
        """Clean text content of each document in place.

        Args:
            documents: List of Document objects to clean.

        Returns:
            The same list of documents with cleaned text.
        """
        for doc in documents:
            doc.text = self._clean_text(doc.text)

        logger.info("Cleaned text in %d documents", len(documents))
        return documents

    def _clean_text(self, text: str) -> str:
        """Apply all cleaning steps to a text string.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        # Strip HTML tags
        text = _HTML_TAG_RE.sub("", text)

        # Handle common encoding artifacts
        text = text.replace("\xa0", " ")  # non-breaking space
        text = text.replace("\u200b", "")  # zero-width space
        text = text.replace("\ufffd", "")  # replacement character

        # Normalize whitespace within lines
        text = _WHITESPACE_RE.sub(" ", text)

        # Normalize excessive newlines
        text = _NEWLINE_RE.sub("\n\n", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        return text.strip()
