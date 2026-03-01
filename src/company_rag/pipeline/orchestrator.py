"""End-to-end RAG pipeline orchestrator."""

import hashlib
import logging

from company_rag.config.settings import Settings, get_settings
from company_rag.embeddings.bge_embedding import BGEEmbedding
from company_rag.generation.base import Response
from company_rag.generation.response_builder import ResponseBuilder
from company_rag.ingestion.base import BaseIngestor
from company_rag.ingestion.earnings_releases import EarningsReleaseIngestor
from company_rag.ingestion.news_releases import NewsReleaseIngestor
from company_rag.ingestion.pdf_files import PDFFileIngestor
from company_rag.ingestion.sec_filings import SECFilingIngestor
from company_rag.ingestion.website import WebsiteIngestor
from company_rag.llm.groq_llm import GroqLLM
from company_rag.llm.ollama_llm import OllamaLLM
from company_rag.llm.openai_llm import OpenAILLM
from company_rag.pipeline.base import BasePipeline
from company_rag.pipeline.dedup import IngestionLog
from company_rag.preprocessing.chunking import SentenceChunker
from company_rag.preprocessing.cleaning import TextCleaner
from company_rag.preprocessing.metadata import MetadataExtractor
from company_rag.retrieval.vector_retriever import VectorRetriever
from company_rag.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

_INGESTOR_REGISTRY: dict[str, type[BaseIngestor]] = {
    "sec_filings": SECFilingIngestor,
    "earnings_releases": EarningsReleaseIngestor,
    "news_releases": NewsReleaseIngestor,
    "website": WebsiteIngestor,
    "pdf": PDFFileIngestor,
}


class RAGPipeline(BasePipeline):
    """End-to-end pipeline: ingest -> chunk -> embed -> store; query -> retrieve -> generate."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the pipeline with all components.

        Args:
            settings: Application settings. Uses default settings if not provided.
        """
        self._settings = settings or get_settings()
        self._ingestors = self._build_ingestors()
        self._cleaner = TextCleaner()
        self._chunker = SentenceChunker()
        self._metadata_extractor = MetadataExtractor()
        self._embedder = BGEEmbedding()
        self._vectorstore = ChromaVectorStore()
        self._retriever = VectorRetriever(
            embedder=self._embedder,
            vectorstore=self._vectorstore,
        )
        self._llm = self._build_llm()
        self._generator = ResponseBuilder(llm=self._llm)
        self._dedup = IngestionLog(db_path=self._settings.dedup_db_path)
        logger.info("RAGPipeline initialized")

    def ingest(self, ticker: str, **kwargs: object) -> dict:
        """Ingest documents for a company through the full pipeline.

        Downloads documents, cleans text, chunks into nodes, generates
        embeddings, and stores in the vector store. Skips previously
        ingested documents using the dedup tracker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            **kwargs: Additional parameters passed to ingestors.

        Returns:
            Dictionary with ingestion statistics.
        """
        stats = {
            "ticker": ticker,
            "documents_fetched": 0,
            "documents_skipped": 0,
            "nodes_created": 0,
            "nodes_stored": 0,
        }

        # Ingest from all sources
        all_documents = []
        for ingestor in self._ingestors:
            documents = ingestor.ingest(ticker, **kwargs)
            all_documents.extend(documents)

        stats["documents_fetched"] = len(all_documents)

        # Deduplicate
        new_documents = []
        for doc in all_documents:
            doc_id = self._get_document_id(doc)
            if self._dedup.is_ingested(doc_id):
                stats["documents_skipped"] += 1
                continue
            new_documents.append(doc)
            self._dedup.mark_ingested(doc_id, doc.metadata)

        if not new_documents:
            logger.info("No new documents to process for %s", ticker)
            return stats

        # Clean -> Chunk -> Metadata
        cleaned = self._cleaner.clean(new_documents)
        nodes = self._chunker.chunk(cleaned)
        nodes = self._metadata_extractor.extract(nodes)
        stats["nodes_created"] = len(nodes)

        # Embed -> Store
        texts = [node.get_content() for node in nodes]
        embeddings = self._embedder.embed(texts)
        self._vectorstore.add(nodes, embeddings)
        stats["nodes_stored"] = len(nodes)

        logger.info(
            "Ingestion complete for %s: %d docs fetched, %d skipped, %d nodes stored",
            ticker,
            stats["documents_fetched"],
            stats["documents_skipped"],
            stats["nodes_stored"],
        )
        return stats

    def query(self, question: str, filters: dict | None = None) -> Response:
        """Ask a question and get a grounded answer with citations.

        Args:
            question: The user's question.
            filters: Optional metadata filters for retrieval.

        Returns:
            Response object with answer, sources, and metadata.
        """
        contexts = self._retriever.retrieve(
            query=question,
            filters=filters,
            top_k=self._settings.retrieval_top_k,
        )
        response = self._generator.generate_response(question, contexts)
        return response

    def _build_ingestors(self) -> list[BaseIngestor]:
        """Build the list of document ingestors based on enabled_sources config.

        Returns:
            List of configured ingestor instances.
        """
        enabled = self._settings.ingestion_enabled_sources
        ingestors = [cls() for name, cls in _INGESTOR_REGISTRY.items() if name in enabled]
        logger.info("Enabled ingestors: %s", [type(i).__name__ for i in ingestors])
        return ingestors

    def _build_llm(self) -> OpenAILLM | OllamaLLM | GroqLLM:
        """Build the LLM provider based on settings.

        Returns:
            Configured LLM instance.
        """
        if self._settings.llm_provider == "ollama":
            return OllamaLLM()
        if self._settings.llm_provider == "groq":
            return GroqLLM()
        return OpenAILLM()

    def _get_document_id(self, doc: object) -> str:
        """Generate a unique document ID for deduplication.

        For SEC filings, uses the accession number. For web sources,
        uses a hash of the URL and content.

        Args:
            doc: Document object with metadata.

        Returns:
            Unique document identifier string.
        """
        metadata = doc.metadata
        accession = metadata.get("accession_number", "")
        if accession:
            return accession

        url = metadata.get("source_url", "")
        content_hash = metadata.get("content_hash", "")
        url_hash = metadata.get("url_hash", "")
        raw = url + content_hash + url_hash
        return hashlib.sha256(raw.encode()).hexdigest()
