"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract interface for text embedding.

    Implementations:
        - BGEEmbedding (bge_embedding.py): BAAI/bge-base-en-v1.5 via sentence-transformers

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseEmbedding
        3. Implement all abstract methods
        4. Register in config/default.yaml under embedding.model_name
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        ...
