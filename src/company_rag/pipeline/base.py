"""Abstract base class for the RAG pipeline orchestrator."""

from abc import ABC, abstractmethod

from company_rag.generation.base import Response


class BasePipeline(ABC):
    """Abstract interface for the RAG pipeline.

    Implementations:
        - RAGPipeline (orchestrator.py): End-to-end pipeline wiring all components

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BasePipeline
        3. Implement all abstract methods
    """

    @abstractmethod
    def ingest(self, ticker: str, **kwargs: object) -> dict:
        """Ingest documents for a company.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Additional ingestion parameters.

        Returns:
            Dictionary with ingestion statistics.
        """
        ...

    @abstractmethod
    def query(self, question: str, filters: dict | None = None) -> Response:
        """Ask a question and get a grounded answer with citations.

        Args:
            question: The user's question.
            filters: Optional metadata filters for retrieval.

        Returns:
            Response object with answer, sources, and metadata.
        """
        ...
