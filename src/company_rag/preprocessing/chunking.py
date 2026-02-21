"""Sentence-based chunking using LlamaIndex SentenceSplitter."""

import logging

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from company_rag.config.settings import get_settings
from company_rag.preprocessing.base import BaseChunker

logger = logging.getLogger(__name__)


class SentenceChunker(BaseChunker):
    """Chunk documents using LlamaIndex's SentenceSplitter.

    Splits text on sentence boundaries with configurable chunk size
    and overlap. Preserves document metadata on each resulting node.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._splitter = SentenceSplitter(
            chunk_size=settings.chunking_chunk_size,
            chunk_overlap=settings.chunking_chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[TextNode]:
        """Split documents into sentence-based text nodes.

        Args:
            documents: List of LlamaIndex Document objects.

        Returns:
            List of TextNode objects with metadata preserved from source documents.
        """
        nodes = self._splitter.get_nodes_from_documents(documents)
        logger.info("Chunked %d documents into %d nodes", len(documents), len(nodes))
        return nodes
