"""SQLite-backed ingestion deduplication tracker."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class IngestionLog:
    """SQLite-backed tracker to prevent re-ingesting the same document.

    Tracks ingested documents by a unique document_id, which is either
    an SEC accession number or a hash of the URL + content for web sources.
    """

    def __init__(self, db_path: str = "./data/ingestion_log.db") -> None:
        """Initialize the ingestion log.

        Args:
            db_path: Path to the SQLite database file.
        """
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._create_table()
        logger.info("Initialized ingestion log at %s", db_path)

    def _create_table(self) -> None:
        """Create the ingestion log table if it doesn't exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_log (
                document_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                source_type TEXT NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
            """
        )
        self._conn.commit()

    def is_ingested(self, document_id: str) -> bool:
        """Check if a document has already been ingested.

        Args:
            document_id: Unique identifier (accession number or URL hash).

        Returns:
            True if the document has been previously ingested.
        """
        cursor = self._conn.execute(
            "SELECT 1 FROM ingestion_log WHERE document_id = ?",
            (document_id,),
        )
        return cursor.fetchone() is not None

    def mark_ingested(self, document_id: str, metadata: dict) -> None:
        """Record a document as ingested.

        Args:
            document_id: Unique identifier (accession number or URL hash).
            metadata: Document metadata to store (ticker, source_type, etc.).
        """
        import json

        self._conn.execute(
            """
            INSERT OR IGNORE INTO ingestion_log (document_id, ticker, source_type, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                document_id,
                metadata.get("ticker", ""),
                metadata.get("source_type", ""),
                json.dumps(metadata),
            ),
        )
        self._conn.commit()
        logger.info("Marked document %s as ingested", document_id)

    def get_ingested_count(self) -> int:
        """Get the total number of ingested documents.

        Returns:
            Count of ingested documents.
        """
        cursor = self._conn.execute("SELECT COUNT(*) FROM ingestion_log")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
