"""Run the RAG pipeline: ingest PDF documents and query them."""

import logging
import sys

from company_rag.pipeline.orchestrator import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Ingest configured PDF documents and start an interactive Q&A loop."""
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Ingest documents
    ticker = pipeline._settings.company_ticker
    logger.info("Ingesting documents for %s...", ticker)
    stats = pipeline.ingest(ticker)
    logger.info("Ingestion stats: %s", stats)

    if stats["nodes_stored"] == 0 and stats["documents_skipped"] == 0:
        logger.warning("No documents were ingested. Check your config/default.yaml settings.")
        logger.warning("Make sure enabled_sources includes 'pdf' and pdf_paths lists your files.")
        sys.exit(1)

    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("RAG Pipeline ready. Ask questions about your documents.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = pipeline.query(question)
        print(f"\nAnswer: {response.answer}\n")
        if response.sources:
            print("Sources:")
            for source in response.sources:
                print(f"  - {source}")
            print()


if __name__ == "__main__":
    main()
