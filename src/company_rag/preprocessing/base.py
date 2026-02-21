"""Abstract base class for text chunking."""

from abc import ABC, abstractmethod

from llama_index.core.schema import Document, TextNode


class BaseChunker(ABC):
    """Abstract interface for document chunking.

    Implementations:
        - SentenceChunker (chunking.py): Wraps LlamaIndex SentenceSplitter

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseChunker
        3. Implement all abstract methods
        4. Register in config/default.yaml under chunking.strategy
    """

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[TextNode]:
        """Split documents into smaller text nodes.

        Args:
            documents: List of LlamaIndex Document objects.

        Returns:
            List of TextNode objects with metadata preserved.
        """
        ...
