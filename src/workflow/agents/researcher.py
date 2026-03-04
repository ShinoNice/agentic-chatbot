from langchain_core.prompts import ChatPromptTemplate

from src.core.config_loader import settings
from src.core.exceptions import LLMResponseError
from src.core.logger import logger
from src.engines.base import BaseLLM

_FALLBACK_ANSWER = (
    "I'm sorry, but I couldn't find any relevant information in the database to answer that."
)
_MIN_ANSWER_LENGTH = 5


class ResearchAgent:
    """Agent that synthesizes retrieved context into a structured, factual response."""

    def __init__(self, llm_engine: BaseLLM):
        self.engine = llm_engine
        self.system_persona = (
            settings.prompts
            .get("agent_system", {})
            .get("researcher", {})
            .get("system_message", "You are a precise Research Assistant.")
        )

    async def generate(self, question: str, documents: list) -> str:
        """Generate a draft answer based strictly on the provided documents."""
        if not documents:
            logger.warning("ResearchAgent called with no documents.")
            return _FALLBACK_ANSWER

        context_blocks = [
            f"--- CONTEXT BLOCK {i + 1} (Source: {doc.metadata.get('source', 'Unknown')}) ---\n"
            f"{doc.page_content}"
            for i, doc in enumerate(documents)
        ]
        full_context = "\n\n".join(context_blocks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                f"{self.system_persona}\n\n"
                "INSTRUCTIONS:\n"
                "- Use ONLY the provided context blocks to answer the question.\n"
                "- If the context does not contain the answer, state that clearly.\n"
                "- Use a professional, objective tone.\n"
                "- Cite your sources within the text using [Source Name] format."
            )),
            ("human",
             "USER QUESTION: {question}\n\nRELEVANT CONTEXT:\n{context}"),
        ])

        try:
            logger.info(
                f"Researcher generating answer for: '{question[:50]}...'")

            messages = prompt.format_messages(
                question=question, context=full_context)
            answer = await self.engine.generate(
                system_prompt=messages[0].content,
                user_prompt=messages[1].content,
            )

            if not answer or len(answer.strip()) < _MIN_ANSWER_LENGTH:
                raise LLMResponseError(
                    "Researcher produced an empty or insufficient response.")

            return answer

        except Exception as e:
            logger.error(f"Research generation failed: {e}", exc_info=True)
            raise LLMResponseError(f"Failed to generate research answer: {e}")
