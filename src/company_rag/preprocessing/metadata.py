"""Metadata extraction and attachment for document chunks."""

import logging

from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

# Required metadata fields that every chunk should have
_REQUIRED_FIELDS = (
    "ticker",
    "source_type",
    "filing_date",
    "section",
    "source_url",
    "document_title",
)


class MetadataExtractor:
    """Attach and validate metadata on text nodes.

    Ensures every chunk has the required metadata fields: ticker, source_type,
    filing_date, section, source_url, and document_title. Missing fields are
    set to empty strings.
    """

    def extract(self, nodes: list[TextNode]) -> list[TextNode]:
        """Validate and enrich metadata on each node.

        Args:
            nodes: List of TextNode objects to process.

        Returns:
            The same list of nodes with validated metadata.
        """
        for node in nodes:
            for field in _REQUIRED_FIELDS:
                if field not in node.metadata:
                    node.metadata[field] = ""

        logger.info("Validated metadata on %d nodes", len(nodes))
        return nodes
