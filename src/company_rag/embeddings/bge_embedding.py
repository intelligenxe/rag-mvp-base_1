"""BGE embedding model wrapper using sentence-transformers."""

import logging

from sentence_transformers import SentenceTransformer

from company_rag.config.settings import get_settings
from company_rag.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class BGEEmbedding(BaseEmbedding):
    """Embedding model using BAAI/bge-base-en-v1.5 via sentence-transformers.

    Produces 768-dimensional normalized embeddings suitable for cosine
    similarity search. Supports batched encoding for efficiency.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.embedding_model_name
        self._device = settings.embedding_device
        self._normalize = settings.embedding_normalize
        self._batch_size = settings.embedding_batch_size

        logger.info("Loading embedding model %s on %s", self._model_name, self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each 768-dimensional).
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        logger.info("Embedded %d texts", len(texts))
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of 768 floats.
        """
        embedding = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()
