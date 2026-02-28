"""Groq LLM provider using the official Groq SDK."""

import logging

from groq import Groq

from company_rag.config.settings import get_settings
from company_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """LLM provider using Groq's fast inference API.

    Runs open-source models (e.g., llama-3.3-70b-versatile, mixtral-8x7b-32768)
    on Groq's hardware. Requires GROQ_API_KEY to be set in the environment
    or .env file.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.llm_groq_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._client = Groq(api_key=settings.groq_api_key)
        logger.info("Initialized Groq LLM with model %s", self._model)

    def generate(self, prompt: str, context: str) -> str:
        """Generate a response using the Groq API.

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
            "Groq generated response (%d chars, model=%s)",
            len(result),
            self._model,
        )
        return result
