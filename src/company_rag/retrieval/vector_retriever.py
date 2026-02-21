"""Vector similarity retriever using embeddings and vector store."""

import logging

from llama_index.core.schema import TextNode

from company_rag.config.settings import get_settings
from company_rag.embeddings.base import BaseEmbedding
from company_rag.retrieval.base import BaseRetriever
from company_rag.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Retrieve documents via cosine similarity search with metadata filtering.

    Embeds the query using the configured embedding model, then queries
    the vector store for the most similar nodes.
    """

    def __init__(self, embedder: BaseEmbedding, vectorstore: BaseVectorStore) -> None:
        """Initialize the vector retriever.

        Args:
            embedder: Embedding model to encode queries.
            vectorstore: Vector store to search against.
        """
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._default_top_k = get_settings().retrieval_top_k

    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[TextNode]:
        """Retrieve relevant text nodes for a query.

        Args:
            query: The search query string.
            filters: Optional metadata filters (e.g., {"source_type": "10-K"}).
            top_k: Number of results to return. Defaults to config value.

        Returns:
            List of relevant TextNode objects ranked by cosine similarity.
        """
        effective_top_k = top_k if top_k != 5 else self._default_top_k

        query_embedding = self._embedder.embed_query(query)
        nodes = self._vectorstore.query(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            filters=filters,
        )

        logger.info("Retrieved %d nodes for query: %.80s...", len(nodes), query)
        return nodes
