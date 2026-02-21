"""Placeholder for future cross-encoder reranking."""

from abc import ABC, abstractmethod

from llama_index.core.schema import TextNode


class BaseReranker(ABC):
    """Abstract interface for result reranking.

    Future implementations:
        - CrossEncoderReranker: BAAI/bge-reranker-base for reranking retrieved results

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseReranker
        3. Implement all abstract methods
    """

    @abstractmethod
    def rerank(self, query: str, nodes: list[TextNode], top_k: int = 5) -> list[TextNode]:
        """Rerank retrieved nodes by relevance to the query.

        Args:
            query: The original search query.
            nodes: List of candidate TextNode objects to rerank.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of TextNode objects.
        """
        ...
