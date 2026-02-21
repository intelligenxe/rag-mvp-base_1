"""Prompt templates for grounded financial Q&A generation."""

from llama_index.core.schema import TextNode

from company_rag.config.settings import get_settings

_NO_INFO_MSG = (
    "I don't have enough information from the available documents to answer this question."
)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a financial research assistant for {company_name} ({ticker}).\n"
    "Answer questions using ONLY the provided context from official "
    "company documents.\n"
    "Rules:\n"
    f'1. If the context does not contain the answer, say "{_NO_INFO_MSG}"\n'
    "2. Always cite the source document (filing type, date, section) "
    "for every claim.\n"
    "3. Never speculate or infer financial figures not explicitly stated "
    "in the context.\n"
    "4. When presenting numbers, include the exact source and "
    "reporting period.\n"
    "5. If information from multiple time periods is present, clearly "
    "distinguish between them."
)

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}"""


def build_system_prompt() -> str:
    """Build the system prompt with company details from settings.

    Returns:
        Formatted system prompt string.
    """
    settings = get_settings()
    return SYSTEM_PROMPT_TEMPLATE.format(
        company_name=settings.company_name,
        ticker=settings.company_ticker,
    )


def format_context(nodes: list[TextNode]) -> str:
    """Format retrieved nodes into a context string for the LLM.

    Each node's text is prefixed with its source metadata for citation purposes.

    Args:
        nodes: List of retrieved TextNode objects.

    Returns:
        Formatted context string with source annotations.
    """
    settings = get_settings()
    citation_format = settings.generation_citation_format

    parts: list[str] = []
    for i, node in enumerate(nodes, 1):
        metadata = node.metadata
        citation = citation_format.format(
            source_type=metadata.get("source_type", "Unknown"),
            filing_date=metadata.get("filing_date", "Unknown"),
            section=metadata.get("section", ""),
        )
        parts.append(f"[Document {i}] {citation}\n{node.get_content()}")

    return "\n\n---\n\n".join(parts)


def build_user_prompt(question: str, nodes: list[TextNode]) -> str:
    """Build the full user prompt with context and question.

    Args:
        question: The user's question.
        nodes: Retrieved context nodes.

    Returns:
        Formatted user prompt string.
    """
    context = format_context(nodes)
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)
