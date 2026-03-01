"""Application settings loaded from YAML config and environment variables."""

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


class Settings(BaseSettings):
    """Application settings loaded from YAML config + environment variables.

    Priority: environment variables > .env file > default.yaml > field defaults
    """

    # Company
    company_ticker: str = "AAPL"
    company_name: str = "Apple Inc."

    # Secrets (from .env)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    hf_token: str = ""

    # Ingestion
    sec_edgar_user_agent: str = "CompanyRAG research@example.com"
    ingestion_filing_types: list[str] = Field(default=["10-K", "10-Q", "8-K"])
    ingestion_years_back: int = 3
    ingestion_download_dir: str = "./data/raw"
    ingestion_website_urls: list[str] = Field(default=[])
    ingestion_pdf_paths: list[str] = Field(default=[])

    # Chunking
    chunking_strategy: str = "sentence"
    chunking_chunk_size: int = 512
    chunking_chunk_overlap: int = 50

    # Embedding
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    embedding_batch_size: int = 32

    # Vector store
    vectorstore_provider: str = "chroma"
    vectorstore_persist_directory: str = "./data/chroma_db"
    vectorstore_collection_name: str = "company_kb"
    vectorstore_distance_metric: str = "cosine"

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_ollama_base_url: str = "http://localhost:11434/v1"
    llm_groq_model: str = "llama-3.3-70b-versatile"

    # Retrieval
    retrieval_strategy: str = "vector"
    retrieval_top_k: int = 5

    # Generation
    generation_include_citations: bool = True
    generation_citation_format: str = "[Source: {source_type}, {filing_date}, {section}]"

    # Dedup
    dedup_db_path: str = "./data/ingestion_log.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings(config_path: str | None = None) -> Settings:
    """Load settings from YAML config file, overlaid with env vars.

    Args:
        config_path: Path to a YAML config file. Defaults to the bundled default.yaml.

    Returns:
        Validated Settings instance.
    """
    yaml_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    yaml_overrides: dict = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
        yaml_overrides = _flatten_yaml(raw)
        logger.info("Loaded config from %s", yaml_path)
    else:
        logger.warning("Config file not found at %s, using defaults", yaml_path)

    return Settings(**yaml_overrides)


def _flatten_yaml(data: dict, prefix: str = "") -> dict:
    """Flatten nested YAML dict into Settings-compatible flat keys.

    Args:
        data: Nested dictionary from YAML.
        prefix: Key prefix for recursion.

    Returns:
        Flat dictionary with underscore-separated keys.
    """
    flat: dict = {}
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_yaml(value, full_key))
        else:
            flat[full_key] = value
    return flat
