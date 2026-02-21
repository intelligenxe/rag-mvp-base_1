"""Abstract base class for retrieval strategies."""

from abc import ABC, abstractmethod

from llama_index.core.schema import TextNode


class BaseRetriever(ABC):
    """Abstract interface for document retrieval.

    Implementations:
        - VectorRetriever (vector_retriever.py): Cosine similarity + metadata filtering

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseRetriever
        3. Implement all abstract methods
        4. Register in config/default.yaml under retrieval.strategy
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[TextNode]:
        """Retrieve relevant text nodes for a query.

        Args:
            query: The search query string.
            filters: Optional metadata filters (e.g., source_type, filing_date range).
            top_k: Number of results to return.

        Returns:
            List of relevant TextNode objects ranked by relevance.
        """
        ...
