# Contributing to company-rag

Thank you for your interest in contributing to company-rag!

## Getting Started

1. Fork the repository
2. Clone your fork and install in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

3. Create a feature branch from `main`

## Development Guidelines

- Follow the conventions in `CLAUDE.md` — it is the source of truth for architecture, code style, and technology choices
- Use absolute imports: `from company_rag.x.y import Z`
- Add type hints to all function signatures
- Use Google-style docstrings on every public class and method
- Use `logging.getLogger(__name__)` instead of `print()`
- Use `pathlib.Path` instead of `os.path`
- No hardcoded values — everything goes through `Settings`

## Adding a New Implementation

Each subpackage defines an abstract base class in `base.py`. To add a new implementation:

1. Create a new file in the appropriate subpackage
2. Inherit from the ABC defined in `base.py`
3. Implement all abstract methods
4. Register the new provider in `config/default.yaml`

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/
ruff format src/
```

## Pull Requests

- Keep PRs focused on a single change
- Write a clear description of what the PR does and why
- Ensure `ruff check` and `ruff format --check` pass before submitting
