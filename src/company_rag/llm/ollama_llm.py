"""Ollama LLM provider using its OpenAI-compatible API."""

import logging

from openai import OpenAI

from company_rag.config.settings import get_settings
from company_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """LLM provider using Ollama's OpenAI-compatible endpoint.

    Connects to a local Ollama instance at the configured base URL.
    Supports models like llama3.1:8b and mistral:7b.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._client = OpenAI(
            base_url=settings.llm_ollama_base_url,
            api_key="ollama",  # Ollama doesn't require a real key
        )
        logger.info(
            "Initialized Ollama LLM with model %s at %s",
            self._model,
            settings.llm_ollama_base_url,
        )

    def generate(self, prompt: str, context: str) -> str:
        """Generate a response using Ollama's OpenAI-compatible API.

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
            "Ollama generated response (%d chars, model=%s)",
            len(result),
            self._model,
        )
        return result
