"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod

from llama_index.core.schema import TextNode


class BaseVectorStore(ABC):
    """Abstract interface for vector storage and retrieval.

    Implementations:
        - ChromaVectorStore (chroma_store.py): ChromaDB with PersistentClient

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseVectorStore
        3. Implement all abstract methods
        4. Register in config/default.yaml under vectorstore.provider
    """

    @abstractmethod
    def add(self, nodes: list[TextNode], embeddings: list[list[float]]) -> None:
        """Add text nodes with their embeddings to the store.

        Args:
            nodes: List of TextNode objects to store.
            embeddings: Corresponding embedding vectors.
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[TextNode]:
        """Query the store for similar nodes.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters (e.g., source_type, filing_date range).

        Returns:
            List of matching TextNode objects ranked by similarity.
        """
        ...

    @abstractmethod
    def delete(self, node_ids: list[str]) -> None:
        """Delete nodes by their IDs.

        Args:
            node_ids: List of node IDs to delete.
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Get collection statistics.

        Returns:
            Dictionary with stats such as total document count.
        """
        ...
