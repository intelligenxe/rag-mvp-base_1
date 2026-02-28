# company-rag

A Python package that builds a Retrieval-Augmented Generation (RAG) knowledge base from the public information of a US NYSE publicly traded company.

## Sources

- SEC filings (10-K, 10-Q, 8-K) via SEC EDGAR
- Earnings releases (8-K Exhibit 99 extraction)
- News/press releases (company newsroom scraping)
- Official company website (curated page scraping)

## Tech Stack

- **LlamaIndex** — Orchestration framework for indexing and retrieval
- **sentence-transformers** (`BAAI/bge-base-en-v1.5`) — Embedding model
- **ChromaDB** — Vector store with persistent storage
- **OpenAI / Ollama** — LLM providers (configurable)
- **Pydantic Settings** — Configuration management (YAML + .env)

## Installation

```bash
pip install -e .
```

For development tools (ruff):

```bash
pip install -e ".[dev]"
```

## Configuration

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

2. Adjust settings in `config/default.yaml` as needed (ticker, model, chunk size, etc.)

## Project Structure

```
src/company_rag/
├── ingestion/        # Data ingestion from SEC EDGAR, newsrooms, websites
├── preprocessing/    # Chunking, metadata extraction, text cleaning
├── embeddings/       # Embedding model wrappers
├── vectorstore/      # Vector store implementations
├── llm/              # LLM provider wrappers
├── retrieval/        # Retrieval strategies
├── generation/       # Prompt templates and response building
├── pipeline/         # End-to-end orchestrator and deduplication
└── config/           # Settings and configuration
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

# APPENDIX A: The logic of the project structure  

The initial project structure is fairly simple and basic but flexible enough so that further or future development may lead to the further enhancement and innovation of features or components.

The future enhancements, refinement and innovation of the various features and components will be carried out in an open source software development setting in GitHub with various contributors working on different parts of the codebase independently and raising PRs independently when they are done implementing their enhancements.

The main code structure needs to be a python Package since it is clearly better for distributed open source development for the following reasons:
1. Isolated development: Each contributor works in their own file with minimal merge conflicts
2. Clear ownership: Easy to assign specific files to specific people
3. Independent testing: Each function has its own test file
4. Easier code review: PRs are focused on single files
5. Versioning & blame: Git history clearly shows who changed what
6. Documentation: Each file can have its own detailed README

[See complete package structure below (includes additional/future modules)]() 

# APPENDIX B: The reason for the abstract base classes




# APPENDIX C: Recommended components and enhancements

## Purpose

The recommended choices below are based on selecting industry-standard, open-source, reliable tools that align with the project's goals: simple initial setup, flexible architecture for future enhancement, and suitability for distributed open-source development on GitHub.

---

## 1. Data Ingestion & Source Specification

### Recommended Choices

| Source | Acquisition Method | Format | Tool |
|---|---|---|---|
| **SEC Filings** (10-K, 10-Q, 8-K, DEF 14A) | SEC EDGAR REST API (free, no API key required) | HTML, XBRL, PDF | **`sec-edgar-downloader`** (MIT license, Python, production-stable since 2021, supports all filing types by ticker or CIK) |
| **Annual Reports** | Usually filed as 10-K on EDGAR; standalone PDFs from investor relations pages | PDF, HTML | `sec-edgar-downloader` for EDGAR versions; **`requests` + `BeautifulSoup`** for IR page scraping |
| **Earnings Releases** | Typically filed as 8-K Exhibit 99.1 on EDGAR | HTML, PDF | `sec-edgar-downloader` (filter `form="8-K"`) |
| **News / Press Releases** | Company newsroom pages | HTML | **`requests` + `BeautifulSoup`** for targeted scraping of known newsroom URLs |
| **Official Company Website** | Curated list of key pages (About, Products, Leadership, etc.) | HTML | **`requests` + `BeautifulSoup`**, or LlamaHub's `SimpleWebPageReader` |

### Why These Choices

- **SEC EDGAR API** is the authoritative, free, public data source for all SEC filings. It requires no API keys — only a User-Agent header identifying your application (SEC fair-access policy). The `sec-edgar-downloader` package wraps this cleanly and is actively maintained (latest release February 2026).
- **`edgartools`** is a strong alternative that adds XBRL parsing and financial statement extraction. For the initial version, `sec-edgar-downloader` is simpler; `edgartools` can be swapped in later for richer structured data extraction.
- Website scraping uses standard Python libraries to keep dependencies minimal. A curated URL list (not a full crawl) avoids legal and scope issues initially.

### Ingestion Frequency

- **Initial version**: Manual/on-demand batch ingestion (run a script to pull all data for a given ticker).
- **Future enhancement**: Scheduled ingestion via cron jobs or a task scheduler (e.g., Celery, APScheduler) with support for incremental updates.

### Package Module

```
src/company_rag/
    ingestion/
        __init__.py
        sec_filings.py       # SEC EDGAR downloader logic
        earnings_releases.py  # 8-K Exhibit 99 extraction
        news_releases.py      # Company newsroom scraper
        website.py            # Company website page loader
        base.py               # Abstract base class for all ingestors
```

---

## 2. Chunking & Preprocessing Strategy

### Recommended Choice: **LlamaIndex's `SentenceSplitter`** (default), with `MarkdownNodeParser` for structured documents

### Configuration Defaults

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 512 tokens | Standard for financial text; balances context richness with retrieval precision |
| Chunk overlap | 50 tokens | Prevents loss of meaning at boundaries |
| Splitter | `SentenceSplitter` (LlamaIndex built-in) | Respects sentence boundaries; avoids mid-sentence cuts |

### Metadata Attached to Each Chunk

Every chunk must carry the following metadata fields (stored alongside the vector):

- `ticker` — Company stock ticker (e.g., "AAPL")
- `source_type` — One of: `10-K`, `10-Q`, `8-K`, `earnings_release`, `news_release`, `website`
- `filing_date` or `publication_date` — ISO 8601 date string
- `section` — If extractable (e.g., "Risk Factors", "MD&A", "Item 1A")
- `source_url` — Original URL or EDGAR accession number
- `document_title` — Human-readable title

### Why This Choice

- LlamaIndex's `SentenceSplitter` is the most widely used chunking strategy in RAG tutorials and production systems. It's battle-tested, requires zero configuration to start, and is easily swappable.
- Financial documents (especially 10-Ks) have well-defined section structures. The architecture should define a `ChunkingStrategy` abstract class so contributors can later implement structure-aware chunking (e.g., parsing 10-K section headers) without modifying the core pipeline.

### Package Module

```
src/company_rag/
    preprocessing/
        __init__.py
        chunking.py          # Chunking strategies (SentenceSplitter default)
        metadata.py          # Metadata extraction and attachment
        cleaning.py          # Text cleaning (HTML strip, whitespace normalization)
        base.py              # Abstract base class for chunking strategies
```

---

## 3. Embedding Model

### Recommended Choice: **`BAAI/bge-base-en-v1.5`** via `sentence-transformers`

| Property | Value |
|---|---|
| Model | `BAAI/bge-base-en-v1.5` |
| Parameters | 109M |
| Dimensions | 768 |
| Max tokens | 512 |
| License | MIT |
| MTEB Ranking | Top tier for its size class |
| Library | `sentence-transformers` (pip install) |

### Why This Choice

- **BGE (BAAI General Embedding)** models are the most widely recommended open-source embedding models for RAG in 2025, consistently ranking at or near the top of MTEB benchmarks for their parameter class.
- `bge-base-en-v1.5` hits the sweet spot: strong retrieval accuracy, fast inference (runs comfortably on CPU for moderate datasets), 768-dimensional vectors (efficient storage), and MIT-licensed.
- It runs locally via `sentence-transformers` with a single line of code — no API keys, no cloud dependency, no cost.
- **Upgrade path**: Contributors can later swap in `bge-large-en-v1.5` (335M params, 1024 dims) for higher accuracy, or `bge-m3` for multilingual support, by changing a single config value.

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(["text chunk here"], normalize_embeddings=True)
```

### Package Module

```
src/company_rag/
    embeddings/
        __init__.py
        embedding_model.py   # Embedding model wrapper with interface
        base.py              # Abstract base class (swap models via config)
```

---

## 4. Vector Store

### Recommended Choice: **ChromaDB**

| Property | Value |
|---|---|
| Type | Open-source vector database |
| License | Apache 2.0 |
| Persistence | Built-in (local disk via DuckDB + Parquet) |
| Metadata filtering | Native support |
| Scale | Comfortable up to ~10M vectors |
| Integration | First-class support in both LlamaIndex and LangChain |
| Install | `pip install chromadb` |

### Why This Choice

- ChromaDB is the industry-standard choice for RAG prototyping and small-to-medium scale applications. It has a developer-friendly Python API, built-in persistence, and native metadata filtering — all critical for this project.
- It stores vectors alongside metadata (ticker, source_type, date, etc.), enabling filtered retrieval queries like "search only within 10-K filings from 2024."
- Zero infrastructure: no Docker, no server, no cloud account needed. It runs embedded in the Python process.
- **Upgrade path**: When the project scales beyond ChromaDB's comfort zone, the abstract vector store interface allows migration to Qdrant (self-hosted, production-grade) or Milvus (distributed, billion-scale) without rewriting the retrieval pipeline.

### Basic Usage

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="company_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)
```

### Package Module

```
src/company_rag/
    vectorstore/
        __init__.py
        chroma_store.py      # ChromaDB implementation
        base.py              # Abstract VectorStore interface
```

---

## 5. LLM Selection & Abstraction

### Recommended Default: **OpenAI `gpt-4o-mini`** (via API), with abstraction for any LLM

### Why This Choice (and why not fully open-source here)

This is the one component where a proprietary API is recommended as the default, for pragmatic reasons:

- For the *generation* step, LLM quality matters enormously — especially for financial data where accuracy is critical. `gpt-4o-mini` offers strong performance at low cost (~$0.15/1M input tokens).
- Open-source LLM alternatives (Llama 3.1 8B, Mistral 7B) require significant GPU resources to run locally, which is a barrier for most contributors.
- The architecture **must** abstract the LLM behind an interface so that any contributor can swap in an open-source model (via Ollama, vLLM, or HuggingFace) by changing a config value.

### Abstraction Design

```python
# config.yaml
llm:
  provider: "openai"          # or "ollama", "huggingface", "anthropic"
  model: "gpt-4o-mini"        # or "llama3.1:8b", "mistral:7b"
  temperature: 0.1
  max_tokens: 1024
```

### Open-Source Alternative for Local Development

| Model | Tool | Notes |
|---|---|---|
| Llama 3.1 8B | **Ollama** (`ollama run llama3.1:8b`) | Free, runs on 8GB+ RAM, good quality |
| Mistral 7B | **Ollama** (`ollama run mistral`) | Lightweight, fast |

**Ollama** is the recommended local LLM runner — it's open-source, cross-platform, and provides an OpenAI-compatible API endpoint, making the swap seamless.

### Package Module

```
src/company_rag/
    llm/
        __init__.py
        openai_llm.py        # OpenAI API wrapper
        ollama_llm.py         # Ollama local model wrapper
        base.py               # Abstract LLM interface
```

---

## 6. Retrieval Strategy

### Recommended Initial Strategy: **Vector similarity search with metadata filtering**

### Configuration Defaults

| Parameter | Value |
|---|---|
| Similarity metric | Cosine similarity |
| Top-K | 5 (retrieve top 5 chunks) |
| Metadata filters | By `source_type`, `filing_date` range, `section` |

### Why This Choice

- Pure vector similarity with metadata filtering is the simplest effective retrieval strategy and the standard starting point for any RAG system.
- ChromaDB's native metadata filtering means you can constrain searches without any additional infrastructure (e.g., "retrieve from 10-K filings only" or "only documents from 2023–2024").

### Future Enhancements (defined as interfaces now, implemented later)

| Enhancement | Description | When to Add |
|---|---|---|
| **Hybrid search (BM25 + vector)** | Combine keyword matching with semantic search using `rank_bm25` | When exact financial terminology matters |
| **Reranking** | Cross-encoder reranker (e.g., `BAAI/bge-reranker-base`) to reorder top-K results | When retrieval precision needs improvement |
| **Query decomposition** | Break complex questions into sub-queries | When handling multi-part financial analysis questions |
| **HyDE (Hypothetical Document Embeddings)** | Generate a hypothetical answer, then search for similar real documents | When queries are vague or conceptual |

### Package Module

```
src/company_rag/
    retrieval/
        __init__.py
        vector_retriever.py  # Default cosine similarity retriever
        hybrid_retriever.py  # Future: BM25 + vector
        reranker.py          # Future: cross-encoder reranking
        base.py              # Abstract Retriever interface
```

---

## 7. Prompting & Response Generation

### System Prompt Strategy

The system prompt must enforce grounded, citation-backed responses for financial data:

```
You are a financial research assistant for {company_name} ({ticker}).
Answer questions using ONLY the provided context from official company documents.

Rules:
1. If the context does not contain the answer, say "I don't have enough information
   from the available documents to answer this question."
2. Always cite the source document (filing type, date, section) for every claim.
3. Never speculate or infer financial figures not explicitly stated in the context.
4. When presenting numbers, include the exact source and reporting period.
5. If information from multiple time periods is present, clearly distinguish between them.
```

### Response Format

- Each response should include inline citations: `[Source: 10-K, 2024-02-15, Risk Factors]`
- Initial version: single-shot Q&A (stateless)
- Future enhancement: multi-turn conversation with chat history

### Package Module

```
src/company_rag/
    generation/
        __init__.py
        prompt_templates.py  # System and user prompt templates
        response_builder.py  # Response formatting with citations
        base.py              # Abstract generator interface
```

---

## 8. Evaluation & Quality Metrics

### Recommended Framework: **RAGAS** (Retrieval Augmented Generation Assessment)

| Property | Value |
|---|---|
| License | Apache 2.0 |
| Install | `pip install ragas` |
| Key feature | Reference-free evaluation (uses LLM-as-judge, no manual labeling needed to start) |
| Integration | Works with LlamaIndex, LangChain, and standalone |

### Metrics to Track

| Metric | What It Measures | Component |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (No hallucination) | Generation |
| **Answer Relevancy** | Is the answer relevant to the question asked? | Generation |
| **Context Precision** | Are the retrieved chunks relevant (signal-to-noise)? | Retrieval |
| **Context Recall** | Was all necessary information retrieved? | Retrieval |

### Benchmark Question Set

Create a manually curated set of 50–100 question-answer pairs covering:

- Factual financial questions (revenue, net income, segment data)
- Risk factor questions
- Strategic/qualitative questions (competitive advantages, market position)
- Time-sensitive questions (comparing year-over-year data)
- Questions that should return "I don't know" (out-of-scope)

### CI Integration

Every PR that modifies retrieval, chunking, or generation code should run the eval suite and report scores. If scores drop below a threshold, the PR is flagged for manual review.

### Package Module

```
src/company_rag/
    evaluation/
        __init__.py
        ragas_eval.py        # RAGAS evaluation runner
        benchmark_data/      # Curated Q&A pairs (JSON/YAML)
        metrics.py           # Custom metrics if needed
```

---

## 9. Configuration & Environment Management

### Recommended Approach: **YAML config file + `.env` for secrets + Pydantic for validation**

### Structure

```
config/
    default.yaml             # Default configuration (committed to repo)
    .env.example             # Template for secrets (committed)
    .env                     # Actual secrets (gitignored)
```

### `default.yaml` Example

```yaml
company:
  ticker: "AAPL"
  name: "Apple Inc."

ingestion:
  sec_edgar_user_agent: "CompanyRAG research@example.com"
  filing_types: ["10-K", "10-Q", "8-K"]
  years_back: 3

chunking:
  strategy: "sentence"
  chunk_size: 512
  chunk_overlap: 50

embedding:
  model: "BAAI/bge-base-en-v1.5"
  device: "cpu"              # or "cuda"

vectorstore:
  provider: "chroma"
  persist_directory: "./data/chroma_db"
  collection_name: "company_kb"

llm:
  provider: "openai"         # or "ollama"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 1024

retrieval:
  strategy: "vector"
  top_k: 5

api:
  host: "0.0.0.0"
  port: 8000
```

### `.env.example`

```
OPENAI_API_KEY=your-key-here
# Optional: for Anthropic, HuggingFace, etc.
ANTHROPIC_API_KEY=
HF_TOKEN=
```

### Validation with Pydantic

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    # ... validated settings
```

### Package Module

```
src/company_rag/
    config/
        __init__.py
        settings.py          # Pydantic settings loader
        default.yaml         # Default config
```

---

## 10. API / Interface Layer

### Recommended Choice: **FastAPI** (REST API) + **Streamlit** (demo UI)

| Component | Tool | License | Purpose |
|---|---|---|---|
| REST API | **FastAPI** | MIT | Programmatic access, production interface |
| Demo UI | **Streamlit** | Apache 2.0 | Quick interactive demo for testing and showcasing |
| CLI | **Click** or **Typer** | BSD / MIT | Developer tooling (ingest, query, evaluate) |

### Why These Choices

- **FastAPI** is the Python standard for REST APIs — async, automatic OpenAPI docs, Pydantic validation built-in. Contributors building frontends, integrations, or mobile apps interact through this.
- **Streamlit** provides a usable demo UI in under 50 lines of code — critical for attracting open-source contributors who want to see the project in action immediately.
- **Typer** (by the FastAPI creator) provides CLI commands for development workflows: `company-rag ingest --ticker AAPL`, `company-rag query "What was revenue in 2024?"`, `company-rag evaluate`.

### API Endpoints (Initial)

```
POST /query          — Ask a question, get a grounded answer with citations
POST /ingest         — Trigger ingestion for a given ticker
GET  /health         — Health check
GET  /collections    — List available company knowledge bases
GET  /stats          — Collection statistics (doc count, last updated)
```

### Package Module

```
src/company_rag/
    api/
        __init__.py
        main.py              # FastAPI app
        routes.py            # Route definitions
        schemas.py           # Request/response Pydantic models
    cli/
        __init__.py
        main.py              # Typer CLI
    ui/
        streamlit_app.py     # Streamlit demo
```

---

## 11. Deployment & Infrastructure

### Recommended Approach: **Docker + docker-compose**

| Component | Tool |
|---|---|
| Containerization | **Docker** (single Dockerfile) |
| Orchestration | **docker-compose** (multi-service: API + UI) |
| Local development | **Python virtual env** (`venv` or `uv`) |

### Dockerfile Strategy

```dockerfile
# Multi-stage build
FROM python:3.11-slim as base
WORKDIR /app
COPY pyproject.toml .
RUN pip install .
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "company_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]
    env_file: .env
  ui:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports: ["8501:8501"]
    depends_on: [api]
```

### Why This Choice

- Docker ensures every contributor, regardless of OS, runs the same environment. This is critical for distributed open-source development.
- `docker-compose` keeps it simple — no Kubernetes knowledge required initially.
- **Upgrade path**: Kubernetes, cloud deployment (AWS ECS, GCP Cloud Run) can be added as the project matures.

---

## 12. Licensing, Contribution Guidelines & CI/CD

### Open-Source License

**Recommended: Apache 2.0**

- Standard for data/ML open-source projects (used by LlamaIndex, ChromaDB, RAGAS, FastAPI ecosystem).
- Permits commercial use, modification, and distribution.
- Includes patent grant (important for enterprise contributors).

### Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, release-ready code |
| `develop` | Integration branch for features |
| `feature/<name>` | Individual feature branches (one per contributor/task) |
| `release/vX.Y.Z` | Release candidates |

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  lint:
    - ruff check src/          # Fast Python linter
    - ruff format --check src/ # Formatting
  test:
    - pytest tests/ --cov=company_rag --cov-report=xml
    - coverage threshold: 80%
  eval:
    - python -m company_rag.evaluation.ragas_eval  # Only on PRs touching core pipeline
```

### Required Files

```
CONTRIBUTING.md       # Contribution guidelines, PR process, code style
CODE_OF_CONDUCT.md    # Contributor Covenant (standard)
README.md             # Project overview, quickstart, architecture diagram
CHANGELOG.md          # Versioned change log
LICENSE               # Apache 2.0
pyproject.toml        # Package metadata, dependencies, build config
```

### Minimum PR Requirements

1. All tests pass
2. Linting passes (ruff)
3. New code has tests in corresponding `tests/` file
4. RAGAS eval scores do not degrade (for pipeline-affecting changes)
5. At least one reviewer approval

---

## 13. Data Freshness & Update Pipeline

### Initial Version: Manual Incremental Ingestion

```bash
# Re-run ingestion for new filings since last run
company-rag ingest --ticker AAPL --since 2024-06-01
```

### Behavior

- New documents are chunked, embedded, and added to the existing ChromaDB collection.
- A simple SQLite metadata table tracks what has already been ingested (accession number for SEC filings, URL + hash for web content) to avoid duplicates.
- No documents are deleted or re-embedded unless explicitly requested.

### Future Enhancement: Scheduled Pipeline

| Component | Tool | Purpose |
|---|---|---|
| Scheduler | **APScheduler** or **cron** | Periodic check for new filings |
| Deduplication | Accession number / URL hash lookup | Prevent re-ingestion |
| Versioning | Timestamped collections or snapshots | Track knowledge base state over time |

### Package Module

```
src/company_rag/
    pipeline/
        __init__.py
        orchestrator.py      # End-to-end: ingest → chunk → embed → store
        scheduler.py         # Future: periodic ingestion
        dedup.py             # Deduplication logic
```

---

## Complete Package Structure

```
company-rag/
├── src/
│   └── company_rag/
│       ├── __init__.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── sec_filings.py
│       │   ├── earnings_releases.py
│       │   ├── news_releases.py
│       │   └── website.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── chunking.py
│       │   ├── metadata.py
│       │   └── cleaning.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── embedding_model.py
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── chroma_store.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── openai_llm.py
│       │   └── ollama_llm.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── vector_retriever.py
│       │   └── reranker.py
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── prompt_templates.py
│       │   └── response_builder.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   └── dedup.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ragas_eval.py
│       │   └── benchmark_data/
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── routes.py
│       │   └── schemas.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       ├── ui/
│       │   └── streamlit_app.py
│       └── config/
│           ├── __init__.py
│           ├── settings.py
│           └── default.yaml
├── tests/
│   ├── test_ingestion/
│   ├── test_preprocessing/
│   ├── test_embeddings/
│   ├── test_vectorstore/
│   ├── test_llm/
│   ├── test_retrieval/
│   ├── test_generation/
│   ├── test_pipeline/
│   ├── test_evaluation/
│   └── test_api/
├── config/
│   ├── default.yaml
│   └── .env.example
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
└── LICENSE
```

---

## Summary of All Choices

| Specification | Choice | License | Why |
|---|---|---|---|
| **Orchestration Framework** | LlamaIndex | MIT | Best for indexing/retrieving from complex document corpora; strong data connector ecosystem |
| **SEC Filing Ingestion** | `sec-edgar-downloader` + EDGAR REST API | MIT | Free, no API key, all filing types, actively maintained |
| **Web Scraping** | `requests` + `BeautifulSoup` | Apache 2.0 / MIT | Standard, minimal dependencies |
| **Chunking** | LlamaIndex `SentenceSplitter` | MIT | Battle-tested default, respects sentence boundaries |
| **Embedding Model** | `BAAI/bge-base-en-v1.5` via `sentence-transformers` | MIT | Top-tier open-source, runs on CPU, 768-dim |
| **Vector Store** | ChromaDB | Apache 2.0 | Zero-config, built-in persistence, metadata filtering |
| **LLM (default)** | OpenAI `gpt-4o-mini` (API) | Proprietary | Quality + cost; abstracted so open-source LLMs (via Ollama) work as drop-in |
| **LLM (local alternative)** | Llama 3.1 8B via Ollama | Meta Community License | Free, local, OpenAI-compatible API |
| **Retrieval** | Cosine similarity + metadata filtering | — | Simplest effective strategy |
| **Evaluation** | RAGAS | Apache 2.0 | Reference-free RAG evaluation, standard metrics |
| **REST API** | FastAPI | MIT | Async, auto-docs, Pydantic validation |
| **Demo UI** | Streamlit | Apache 2.0 | Minimal code for interactive demo |
| **CLI** | Typer | MIT | FastAPI creator's CLI framework |
| **Config** | YAML + `.env` + Pydantic | — | Industry standard for Python projects |
| **Containerization** | Docker + docker-compose | Apache 2.0 | Reproducible environments across contributors |
| **CI/CD** | GitHub Actions | — | Native to GitHub; free for open-source |
| **Linter** | Ruff | MIT | Fastest Python linter, replaces flake8+isort+black |
| **Testing** | pytest + pytest-cov | MIT | Python testing standard |
| **License** | Apache 2.0 | — | Standard for ML/data open-source projects |


