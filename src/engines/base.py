from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract base class for LLM engines (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str | None = None,
    ) -> str:
        """Generate a text response from the LLM."""
        ...

    @abstractmethod
    def get_model(self) -> Any:
        """Return the underlying model instance (e.g. for `.with_structured_output()`)."""
        ...
