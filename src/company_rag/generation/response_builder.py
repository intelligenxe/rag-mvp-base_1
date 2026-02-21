"""Response builder that assembles grounded answers with inline citations."""

import logging

from llama_index.core.schema import TextNode

from company_rag.generation.base import BaseGenerator, Response
from company_rag.generation.prompt_templates import build_system_prompt, build_user_prompt
from company_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class ResponseBuilder(BaseGenerator):
    """Generate grounded responses with inline citations.

    Uses the configured LLM to generate answers from retrieved context,
    formatting the system and user prompts to enforce citation and grounding rules.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """Initialize the response builder.

        Args:
            llm: LLM provider to use for generation.
        """
        self._llm = llm

    def generate_response(self, query: str, contexts: list[TextNode]) -> Response:
        """Generate a grounded response from query and retrieved contexts.

        Args:
            query: The user's question.
            contexts: Retrieved TextNode objects to use as context.

        Returns:
            Response object with answer, source nodes, and metadata.
        """
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query, contexts)

        answer = self._llm.generate(prompt=user_prompt, context=system_prompt)

        logger.info("Generated response (%d chars) for query: %.80s...", len(answer), query)

        return Response(
            answer=answer,
            source_nodes=contexts,
            metadata={
                "query": query,
                "num_contexts": len(contexts),
            },
        )
