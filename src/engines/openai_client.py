from langchain_openai import ChatOpenAI

from src.core.config_loader import settings
from src.core.logger import logger
from src.engines.base import BaseLLM


class OpenAIClient(BaseLLM):
    """OpenAI Chat model implementation of BaseLLM."""

    def __init__(self, model_name: str | None = None, temp: float | None = None):
        self.model_name = model_name or settings.llm.primary_llm
        self.temperature = temp if temp is not None else settings.llm.temperature

        logger.info(f"Initializing OpenAIClient with model: {self.model_name}")

        self._model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm.max_tokens,
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str | None = None,
    ) -> str:
        """Generate a text response, optionally overriding the model for this call."""
        model = self._model
        if model_name and model_name != self.model_name:
            model = ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                api_key=settings.openai_api_key,
            )

        try:
            response = await model.ainvoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            return response.content
        except Exception as e:
            logger.error(f"OpenAI Generation Error: {e}", exc_info=True)
            raise

    def get_model(self) -> ChatOpenAI:
        """Expose the raw ChatOpenAI instance (e.g. for `.with_structured_output()`)."""
        return self._model
