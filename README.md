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
