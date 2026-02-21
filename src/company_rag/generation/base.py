"""Abstract base class for response generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llama_index.core.schema import TextNode


@dataclass
class Response:
    """Structured response from the generation component.

    Attributes:
        answer: The generated answer text with inline citations.
        source_nodes: The context nodes used to generate the answer.
        metadata: Additional response metadata (e.g., model used, token counts).
    """

    answer: str
    source_nodes: list[TextNode]
    metadata: dict = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract interface for response generation.

    Implementations:
        - ResponseBuilder (response_builder.py): Assembles answer with inline citations

    To add a new implementation:
        1. Create a new file in this directory
        2. Inherit from BaseGenerator
        3. Implement all abstract methods
    """

    @abstractmethod
    def generate_response(self, query: str, contexts: list[TextNode]) -> Response:
        """Generate a grounded response from query and retrieved contexts.

        Args:
            query: The user's question.
            contexts: Retrieved TextNode objects to use as context.

        Returns:
            Response object with answer, sources, and metadata.
        """
        ...
