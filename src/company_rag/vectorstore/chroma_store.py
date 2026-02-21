"""ChromaDB vector store implementation."""

import logging

import chromadb
from llama_index.core.schema import TextNode

from company_rag.config.settings import get_settings
from company_rag.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by ChromaDB with persistent storage.

    Uses ChromaDB's PersistentClient with cosine similarity for storage
    and retrieval of text node embeddings with metadata filtering.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.vectorstore_persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=settings.vectorstore_collection_name,
            metadata={"hnsw:space": settings.vectorstore_distance_metric},
        )
        logger.info(
            "Initialized ChromaDB collection '%s' at %s",
            settings.vectorstore_collection_name,
            settings.vectorstore_persist_directory,
        )

    def add(self, nodes: list[TextNode], embeddings: list[list[float]]) -> None:
        """Add text nodes with their embeddings to ChromaDB.

        Args:
            nodes: List of TextNode objects to store.
            embeddings: Corresponding embedding vectors.
        """
        if not nodes:
            return

        ids = [node.node_id for node in nodes]
        documents = [node.get_content() for node in nodes]
        metadatas = [self._sanitize_metadata(node.metadata) for node in nodes]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Added %d nodes to ChromaDB", len(nodes))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[TextNode]:
        """Query ChromaDB for similar nodes with optional metadata filtering.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters. Supports:
                - {"source_type": "10-K"} for exact match
                - {"filing_date_gte": "2024-01-01"} for range filters

        Returns:
            List of matching TextNode objects ranked by cosine similarity.
        """
        query_params: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filters:
            where = self._build_where_filter(filters)
            if where:
                query_params["where"] = where

        results = self._collection.query(**query_params)

        nodes: list[TextNode] = []
        if results["ids"] and results["ids"][0]:
            for i, node_id in enumerate(results["ids"][0]):
                text = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                node = TextNode(text=text, id_=node_id, metadata=metadata)
                nodes.append(node)

        logger.info("Query returned %d results", len(nodes))
        return nodes

    def delete(self, node_ids: list[str]) -> None:
        """Delete nodes from ChromaDB by their IDs.

        Args:
            node_ids: List of node IDs to delete.
        """
        if not node_ids:
            return

        self._collection.delete(ids=node_ids)
        logger.info("Deleted %d nodes from ChromaDB", len(node_ids))

    def get_stats(self) -> dict:
        """Get ChromaDB collection statistics.

        Returns:
            Dictionary with collection name and document count.
        """
        count = self._collection.count()
        return {
            "collection_name": self._collection.name,
            "document_count": count,
        }

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata values for ChromaDB compatibility.

        ChromaDB only supports str, int, float, and bool metadata values.

        Args:
            metadata: Raw metadata dictionary.

        Returns:
            Sanitized metadata with only ChromaDB-compatible types.
        """
        sanitized: dict = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized

    def _build_where_filter(self, filters: dict) -> dict | None:
        """Build a ChromaDB where filter from a user-friendly filter dict.

        Args:
            filters: Filter dictionary with keys like "source_type", "filing_date_gte".

        Returns:
            ChromaDB-compatible where filter, or None if no valid filters.
        """
        conditions: list[dict] = []
        for key, value in filters.items():
            if key.endswith("_gte"):
                field = key[:-4]
                conditions.append({field: {"$gte": value}})
            elif key.endswith("_lte"):
                field = key[:-4]
                conditions.append({field: {"$lte": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
