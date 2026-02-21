# CLAUDE.md — Company RAG Knowledge Base

## Project Identity

This is **company-rag**, an open-source Python package that builds a Retrieval-Augmented Generation (RAG) knowledge base from the public information of a US NYSE publicly traded company. Sources include SEC filings (10-K, 10-Q, 8-K), earnings releases, news/press releases, and the official company website.

The project is designed for distributed open-source development on GitHub, where multiple contributors work on different modules independently and raise PRs for their enhancements.

---

## Architecture Principles

Follow these principles in every file you create or modify:

1. **Package structure is mandatory.** All source code lives under `src/company_rag/` as a proper Python package. Each logical component is its own subpackage with its own `__init__.py`. This is non-negotiable — it enables isolated development, clear ownership, independent testing, and clean git blame.

2. **Abstract base classes everywhere.** Every subpackage has a `base.py` defining abstract interfaces (using `abc.ABC` and `@abstractmethod`). Concrete implementations inherit from these. This allows contributors to swap implementations (e.g., a new vector store, a new chunking strategy) by adding a single file without touching existing code.

3. **Configuration-driven behavior.** No hardcoded model names, paths, API keys, or parameters. Everything is controlled through `config/default.yaml` (committed) and `.env` (gitignored, for secrets). Use Pydantic (`pydantic-settings`) for validation. A contributor should be able to switch from OpenAI to Ollama, or from ChromaDB to Qdrant, by changing a config value.

4. **Start simple, design for extension.** Implement the simplest working version first. But always define the interface broadly enough that future enhancements (hybrid search, reranking, multi-turn chat, scheduled ingestion) can be added as new files without modifying existing ones. Open/Closed Principle.

---

## Technology Stack — Mandatory Choices

Do NOT substitute these without explicit instruction. These were selected for reliability, open-source licensing, community adoption, and alignment with the project's distributed development model.

### Core Framework
- **LlamaIndex** — Primary orchestration framework for indexing and retrieval
- Use LlamaIndex's node/document abstractions, `SentenceSplitter`, and query engine patterns
- Import from `llama_index.core` (v0.10+ namespace)

### Data Ingestion
- **`sec-edgar-downloader`** (MIT) — SEC EDGAR filings by ticker or CIK
  - SEC EDGAR API is free, requires no API key, only a User-Agent header
  - User-Agent format: `"CompanyRAG research@example.com"` (configurable)
  - Supports all filing types: 10-K, 10-Q, 8-K, DEF 14A, etc.
- **`requests` + `beautifulsoup4`** — Company website and newsroom scraping
  - Scrape a curated list of URLs, not a full crawl
- **`unstructured`** or `llama_index.readers` — PDF and HTML text extraction

### Chunking & Preprocessing
- **`SentenceSplitter`** from `llama_index.core.node_parser` — Default chunking strategy
  - `chunk_size=512`, `chunk_overlap=50`
- Attach metadata to every chunk: `ticker`, `source_type`, `filing_date`, `section`, `source_url`, `document_title`
- Text cleaning: strip HTML tags, normalize whitespace, handle encoding issues

### Embedding Model
- **`BAAI/bge-base-en-v1.5`** via `sentence-transformers`
  - 109M parameters, 768 dimensions, MIT license
  - Runs on CPU by default, CUDA optional
  - Always normalize embeddings: `model.encode(texts, normalize_embeddings=True)`
  - Wrap in a class implementing the abstract `BaseEmbedding` interface

### Vector Store
- **ChromaDB** (Apache 2.0)
  - Use `PersistentClient` with configurable path (default: `./data/chroma_db`)
  - Collection uses cosine similarity: `metadata={"hnsw:space": "cosine"}`
  - Store chunk text, embedding, and all metadata fields
  - Use native metadata filtering for retrieval constraints (source_type, date range)

### LLM
- **Default: OpenAI `gpt-4o-mini`** via `openai` Python SDK
  - Temperature: 0.1 (low creativity for factual financial answers)
  - max_tokens: 1024
- **Alternative: Ollama** for local/free development
  - Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`
  - Models: `llama3.1:8b`, `mistral:7b`
- **Both must implement the same `BaseLLM` abstract interface** so they are interchangeable via config

### Retrieval
- **Default: Vector similarity search** (cosine) with ChromaDB metadata filtering
  - Top-K: 5 (configurable)
  - Support filtering by: `source_type`, `filing_date` range, `section`
- **Future (interfaces defined now, not implemented):** hybrid BM25+vector, cross-encoder reranking (`BAAI/bge-reranker-base`), query decomposition, HyDE

### Generation & Prompting
- System prompt enforces grounded, citation-backed responses:
  ```
  You are a financial research assistant for {company_name} ({ticker}).
  Answer questions using ONLY the provided context from official company documents.
  Rules:
  1. If the context does not contain the answer, say "I don't have enough information from the available documents to answer this question."
  2. Always cite the source document (filing type, date, section) for every claim.
  3. Never speculate or infer financial figures not explicitly stated in the context.
  4. When presenting numbers, include the exact source and reporting period.
  5. If information from multiple time periods is present, clearly distinguish between them.
  ```
- Response format includes inline citations: `[Source: 10-K, 2024-02-15, Risk Factors]`
- Initial version: single-shot Q&A (stateless). No chat history yet.

### Configuration
- **`config/default.yaml`** — All defaults (committed to repo)
- **`.env`** — Secrets only: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN` (gitignored)
- **`.env.example`** — Template with placeholder values (committed)
- **Pydantic `BaseSettings`** in `src/company_rag/config/settings.py` — Loads YAML + .env, validates types

### Data Freshness & Deduplication
- SQLite metadata table (`data/ingestion_log.db`) tracks ingested documents by accession number (SEC) or URL hash (web)
- Supports incremental ingestion by date
- No automatic deletion or re-embedding unless explicitly requested

---

## Package Structure

Generate exactly this structure. Every file listed here must exist.

```
company-rag/
├── src/
│   └── company_rag/
│       ├── __init__.py                      # Package version: __version__ = "0.1.0"
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── base.py                      # ABC: BaseIngestor with methods: ingest(ticker, **kwargs) -> List[Document]
│       │   ├── sec_filings.py               # SECFilingIngestor — uses sec-edgar-downloader
│       │   ├── earnings_releases.py          # EarningsReleaseIngestor — 8-K Exhibit 99 extraction
│       │   ├── news_releases.py              # NewsReleaseIngestor — company newsroom scraper
│       │   └── website.py                    # WebsiteIngestor — curated page scraper
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── base.py                      # ABC: BaseChunker with methods: chunk(documents) -> List[Node]
│       │   ├── chunking.py                  # SentenceChunker — wraps LlamaIndex SentenceSplitter
│       │   ├── metadata.py                  # MetadataExtractor — attaches ticker, source_type, date, section, url, title
│       │   └── cleaning.py                  # TextCleaner — HTML strip, whitespace normalize, encoding fix
│       │
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py                      # ABC: BaseEmbedding with methods: embed(texts) -> List[List[float]]
│       │   └── bge_embedding.py             # BGEEmbedding — wraps sentence-transformers BAAI/bge-base-en-v1.5
│       │
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── base.py                      # ABC: BaseVectorStore with methods: add, query, delete, get_stats
│       │   └── chroma_store.py              # ChromaVectorStore — wraps chromadb PersistentClient
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py                      # ABC: BaseLLM with methods: generate(prompt, context) -> str
│       │   ├── openai_llm.py                # OpenAILLM — wraps openai.ChatCompletion
│       │   └── ollama_llm.py                # OllamaLLM — calls Ollama's OpenAI-compatible endpoint
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── base.py                      # ABC: BaseRetriever with methods: retrieve(query, filters, top_k) -> List[Node]
│       │   ├── vector_retriever.py          # VectorRetriever — cosine similarity + metadata filtering via ChromaDB
│       │   └── reranker.py                  # Placeholder: BaseReranker ABC + stub for future cross-encoder
│       │
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── base.py                      # ABC: BaseGenerator with methods: generate_response(query, contexts) -> Response
│       │   ├── prompt_templates.py          # SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context()
│       │   └── response_builder.py          # ResponseBuilder — assembles answer with inline citations
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── orchestrator.py              # RAGPipeline — end-to-end: ingest → chunk → embed → store; query → retrieve → generate
│       │   └── dedup.py                     # IngestionLog — SQLite-backed deduplication tracker
│       │
│       └── config/
│           ├── __init__.py
│           ├── settings.py                  # Pydantic Settings class, loads YAML + .env
│           └── default.yaml                 # All default configuration values
│
├── config/
│   ├── default.yaml                         # Copy of src/company_rag/config/default.yaml (convenience)
│   └── .env.example
│
├── data/                                    # Gitignored except .gitkeep
│   ├── .gitkeep
│   ├── raw/                                 # Downloaded raw filings/pages
│   ├── chroma_db/                           # ChromaDB persistence
│   └── ingestion_log.db                     # SQLite dedup tracker
│
├── pyproject.toml
├── .gitignore
├── .env.example
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
└── LICENSE                                  # Apache 2.0
```

---

## Code Style & Conventions

### Python
- **Python 3.11+** — Use modern typing: `list[str]` not `List[str]`, `X | None` not `Optional[X]`
- **Type hints on all function signatures** — parameters and return types
- **Docstrings** — Google style on every public class and method
- **No star imports** — Always explicit: `from company_rag.ingestion.base import BaseIngestor`
- **Ruff** for linting and formatting — configured in `pyproject.toml`
- **Logging** — Use `logging.getLogger(__name__)` in every module. Never `print()`.
- **No hardcoded values** — Everything comes from `Settings` (which loads config/default.yaml + .env)

### Abstract Base Classes
Every `base.py` follows this pattern:
```python
from abc import ABC, abstractmethod

class BaseFoo(ABC):
    """Abstract interface for Foo component.

    Implementations:
        - ConcreteFoo (concrete_foo.py): Description

    To add a new implementation:
        1. Create new_foo.py in this directory
        2. Inherit from BaseFoo
        3. Implement all abstract methods
        4. Register in config/default.yaml under foo.provider
    """

    @abstractmethod
    def do_something(self, input: str) -> str:
        """One-line description.

        Args:
            input: Description.

        Returns:
            Description.
        """
        ...
```

### Imports Within Package
Always use absolute imports from the package root:
```python
# YES
from company_rag.config.settings import get_settings
from company_rag.ingestion.base import BaseIngestor

# NO
from .base import BaseIngestor
from ..config.settings import get_settings
```

---

## pyproject.toml Specification

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "company-rag"
version = "0.1.0"
description = "RAG knowledge base for US NYSE publicly traded companies"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.11"
dependencies = [
    "llama-index-core>=0.10",
    "llama-index-readers-file>=0.1",
    "sentence-transformers>=2.2",
    "chromadb>=0.4",
    "openai>=1.0",
    "sec-edgar-downloader>=5.0",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "lxml>=4.9",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "RUF"]
```

---

## config/default.yaml — Full Default Configuration

```yaml
company:
  ticker: "AAPL"
  name: "Apple Inc."

ingestion:
  sec_edgar_user_agent: "CompanyRAG research@example.com"
  filing_types:
    - "10-K"
    - "10-Q"
    - "8-K"
  years_back: 3
  download_dir: "./data/raw"
  website_urls: []  # Curated list of company page URLs to scrape

chunking:
  strategy: "sentence"             # "sentence" (only option initially)
  chunk_size: 512
  chunk_overlap: 50

embedding:
  model_name: "BAAI/bge-base-en-v1.5"
  device: "cpu"                    # "cpu" or "cuda"
  normalize: true
  batch_size: 32

vectorstore:
  provider: "chroma"               # "chroma" (only option initially)
  persist_directory: "./data/chroma_db"
  collection_name: "company_kb"
  distance_metric: "cosine"

llm:
  provider: "openai"               # "openai" or "ollama"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 1024
  # Ollama-specific
  ollama_base_url: "http://localhost:11434/v1"

retrieval:
  strategy: "vector"               # "vector" (only option initially)
  top_k: 5

generation:
  include_citations: true
  citation_format: "[Source: {source_type}, {filing_date}, {section}]"

dedup:
  db_path: "./data/ingestion_log.db"
```

---

## Settings Loader Pattern

`src/company_rag/config/settings.py` must implement this pattern:

```python
import yaml
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

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
    hf_token: str = ""

    # ... all other settings with defaults matching default.yaml

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

@lru_cache()
def get_settings(config_path: str | None = None) -> Settings:
    """Load settings from YAML config file, overlaid with env vars."""
    # Load YAML, create Settings with YAML values as overrides
    ...
```

All modules obtain settings via:
```python
from company_rag.config.settings import get_settings
settings = get_settings()
```

---

## Key Implementation Details

### Ingestion (sec_filings.py)
```python
from sec_edgar_downloader import Downloader

class SECFilingIngestor(BaseIngestor):
    def ingest(self, ticker: str, filing_types: list[str], years_back: int, ...) -> list[Document]:
        dl = Downloader(company_name="CompanyRAG", email_address=settings.sec_edgar_user_agent)
        for filing_type in filing_types:
            dl.get(filing_type, ticker, limit=years_back * 4)  # approximate
        # Parse downloaded files, extract text, create Document objects with metadata
        ...
```

### Deduplication (dedup.py)
```python
import sqlite3

class IngestionLog:
    """SQLite-backed tracker to prevent re-ingesting the same document."""

    def __init__(self, db_path: str = "./data/ingestion_log.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def is_ingested(self, document_id: str) -> bool: ...
    def mark_ingested(self, document_id: str, metadata: dict) -> None: ...
    # document_id = accession_number (SEC) or sha256(url + content_hash) (web)
```

### Orchestrator (orchestrator.py)
```python
class RAGPipeline:
    """End-to-end pipeline: ingest → chunk → embed → store → query → retrieve → generate."""

    def __init__(self, settings: Settings):
        self.ingestors = self._build_ingestors(settings)
        self.chunker = self._build_chunker(settings)
        self.embedder = self._build_embedder(settings)
        self.vectorstore = self._build_vectorstore(settings)
        self.retriever = self._build_retriever(settings)
        self.llm = self._build_llm(settings)
        self.generator = self._build_generator(settings)
        self.dedup = IngestionLog(settings.dedup_db_path)

    def ingest(self, ticker: str, **kwargs) -> dict:
        """Ingest documents for a company. Returns stats dict."""
        ...

    def query(self, question: str, filters: dict | None = None) -> QueryResponse:
        """Ask a question, get a grounded answer with citations."""
        contexts = self.retriever.retrieve(question, filters=filters, top_k=settings.retrieval_top_k)
        response = self.generator.generate_response(question, contexts)
        return response
```

---

## .gitignore Essentials

```
# Environment
.env
*.pyc
__pycache__/
.venv/
venv/

# Data
data/chroma_db/
data/raw/
data/ingestion_log.db
!data/.gitkeep

# IDE
.vscode/
.idea/

# Build
dist/
build/
*.egg-info/

# OS
.DS_Store
Thumbs.db
```

---

## Task Execution Order

When asked to generate the project, create files in this order:

1. **Project scaffolding first:** `pyproject.toml`, `LICENSE`, `.gitignore`, `.env.example`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`
2. **Config layer:** `config/default.yaml`, `src/company_rag/config/settings.py`
3. **Abstract interfaces (all `base.py` files)** — These define the contracts before any implementation
4. **Core implementations in dependency order:**
   - `ingestion/` (no internal deps)
   - `preprocessing/` (depends on ingestion output)
   - `embeddings/` (no internal deps)
   - `vectorstore/` (depends on embeddings)
   - `llm/` (no internal deps)
   - `retrieval/` (depends on vectorstore)
   - `generation/` (depends on llm)
   - `pipeline/` (depends on all above — this is the orchestrator)

---

## What NOT to Do

- **Do NOT use LangChain** — This project uses LlamaIndex as the orchestration framework. Do not mix frameworks.
- **Do NOT hardcode API keys, model names, file paths, or any parameters** — Everything goes through `Settings`.
- **Do NOT create monolithic files** — Each class gets its own file. This is a package, not a script.
- **Do NOT use relative imports** — Always `from company_rag.x.y import Z`.
- **Do NOT skip type hints or docstrings** — Every public function and class is documented and typed.
- **Do NOT implement future enhancements** (hybrid search, reranking, multi-turn chat, scheduled ingestion) — Only define their abstract interfaces. Implementation comes later via contributor PRs.
- **Do NOT use `print()`** — Use `logging.getLogger(__name__)`.
- **Do NOT use `os.path`** — Use `pathlib.Path`.
- **Do NOT install packages not listed in pyproject.toml** — If you need something, add it to the dependency list.
