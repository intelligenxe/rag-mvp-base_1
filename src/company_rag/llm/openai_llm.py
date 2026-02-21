"""OpenAI LLM provider."""

import logging

from openai import OpenAI

from company_rag.config.settings import get_settings
from company_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """LLM provider using the OpenAI API.

    Defaults to gpt-4o-mini with low temperature for factual financial answers.
    Requires OPENAI_API_KEY to be set in the environment or .env file.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._client = OpenAI(api_key=settings.openai_api_key)
        logger.info("Initialized OpenAI LLM with model %s", self._model)

    def generate(self, prompt: str, context: str) -> str:
        """Generate a response using the OpenAI Chat Completions API.

        Args:
            prompt: The user's question or instruction.
            context: Retrieved context to ground the response.

        Returns:
            Generated text response.
        """
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        result = response.choices[0].message.content or ""
        logger.info(
            "OpenAI generated response (%d chars, model=%s)",
            len(result),
            self._model,
        )
        return result
