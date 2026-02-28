"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract interface for language model providers.

    Implementations:
        - OpenAILLM (openai_llm.py): OpenAI API (gpt-4o-mini default)
        - OllamaLLM (ollama_llm.py): Ollama's OpenAI-compatible endpoint
        - GroqLLM (groq_llm.py): Groq API with Llama models

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseLLM
        3. Implement all abstract methods
        4. Register in config/default.yaml under llm.provider
    """

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        """Generate a response given a prompt and context.

        Args:
            prompt: The user's question or instruction.
            context: Retrieved context to ground the response.

        Returns:
            Generated text response.
        """
        ...
