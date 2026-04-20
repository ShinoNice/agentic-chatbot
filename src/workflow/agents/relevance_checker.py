from langchain_core.prompts import ChatPromptTemplate

from src.core.config_loader import settings
from src.core.exceptions import RelevanceAuditError
from src.core.logger import logger
from src.engines.base import BaseLLM
from src.schemas.agent_schemas import RelevanceResponse, RelevanceStatus


class RelevanceCheckerAgent:
    """Agent that audits retrieved documents for relevance before answer generation."""

    def __init__(self, llm_engine: BaseLLM):
        self.model = llm_engine.get_model()
        self.system_persona = (
            settings.prompts.get("agent_system", {})
            .get("relevance_checker", {})
            .get("system_message", "You are a strict Relevance Auditor.")
        )

        try:
            self.structured_llm = self.model.with_structured_output(RelevanceResponse)
        except Exception as e:
            logger.error(f"Failed to bind RelevanceResponse schema: {e}")
            raise RelevanceAuditError("LLM provider does not support structured output.")

    async def check(self, question: str, documents: list) -> RelevanceStatus:
        """Evaluate whether the documents contain enough info to answer the question."""
        if not documents:
            logger.warning("RelevanceCheckerAgent received zero documents.")
            return RelevanceStatus.NO_MATCH

        context_text = "\n\n".join(f"Doc {i}: {d.page_content}" for i, d in enumerate(documents))

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        f"{self.system_persona}\n\n"
                        "Rules:\n"
                        "1. If the context directly answers the question, return CAN_ANSWER.\n"
                        "2. If the context mentions parts of the topic but lacks key details, return PARTIAL.\n"
                        "3. If the context is unrelated, return NO_MATCH.\n"
                        "Be objective. If the data isn't there, do not assume it is."
                    ),
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )

        try:
            chain = prompt | self.structured_llm
            result: RelevanceResponse = await chain.ainvoke(
                {
                    "question": question,
                    "context": context_text[: settings.agent.max_verifier_context_chars],
                }
            )

            logger.info(f"Relevance Audit: {result.status} | Reason: {result.reasoning}")
            return result.status

        except Exception as e:
            logger.error(f"Error during relevance audit: {e}", exc_info=True)
            return RelevanceStatus.NO_MATCH
