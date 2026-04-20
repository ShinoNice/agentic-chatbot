from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import OpenAIEmbeddings

from src.core.config_loader import settings
from src.core.logger import logger


class BaseEmbeddingModel(ABC):
    """Abstract interface for all embedding providers."""

    @abstractmethod
    def get_embeddings(self) -> Any:
        """Return the underlying LangChain embedding object."""
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI implementation for text embeddings."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.llm.embedding_model
        logger.info(f"Initializing OpenAI Embeddings with model: {self.model_name}")

        try:
            self._embeddings = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=settings.openai_api_key,
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Embeddings: {e}")
            raise

    def get_embeddings(self) -> OpenAIEmbeddings:
        return self._embeddings


def get_embedding_engine() -> BaseEmbeddingModel:
    """Return the configured embedding engine (defaults to OpenAI)."""
    return OpenAIEmbeddingModel()
