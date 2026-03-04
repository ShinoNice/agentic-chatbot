from langchain_core.prompts import ChatPromptTemplate

from src.core.exceptions import AgentError
from src.core.logger import logger
from src.engines.base import BaseLLM
from src.schemas.agent_schemas import VerificationReport

_MAX_CONTEXT_CHARS = 12_000


class VerificationAgent:
    """Agent that audits a draft answer against source context for hallucinations and contradictions."""

    def __init__(self, llm_engine: BaseLLM):
        self.model = llm_engine.get_model()

        try:
            self.structured_llm = self.model.with_structured_output(
                VerificationReport)
        except Exception as e:
            logger.error(f"Failed to bind VerificationReport schema: {e}")
            raise AgentError(
                "LLM provider does not support structured output.")

    async def verify(self, question: str, answer: str, documents: list) -> VerificationReport:
        """Compare the draft answer against the provided documents."""
        if not answer:
            logger.warning("VerificationAgent received an empty draft answer.")
            return VerificationReport(supported=False, relevant=False)

        context_text = "\n\n".join(
            f"SOURCE [{d.metadata.get('source', 'Unknown')}]: {d.page_content}"
            for d in documents
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a strict QA Auditor. Your task is to verify an AI-generated answer "
                "against the provided search context.\n\n"
                "Audit Instructions:\n"
                "1. Check if every factual claim in the answer is backed by the context.\n"
                "2. Identify 'Unsupported Claims' (claims not mentioned in context).\n"
                "3. Identify 'Contradictions' (claims that conflict with the context).\n"
                "4. Assess if the answer actually addresses the user's question.\n\n"
                "Be pedantic. If the context says 'maybe' and the answer says 'definitely', "
                "mark it as unsupported."
            )),
            ("human", (
                "USER QUESTION: {question}\n\n"
                "DRAFT ANSWER TO AUDIT:\n{answer}\n\n"
                "SOURCE CONTEXT:\n{context}"
            )),
        ])

        try:
            chain = prompt | self.structured_llm
            report: VerificationReport = await chain.ainvoke({
                "question": question,
                "answer": answer,
                "context": context_text[:_MAX_CONTEXT_CHARS],
            })

            if not report.supported:
                logger.warning(
                    f"Hallucination detected - Unsupported: {len(report.unsupported_claims)} | "
                    f"Contradictions: {len(report.contradictions)}"
                )
            else:
                logger.info("Answer verified: no hallucinations found.")

            return report

        except Exception as e:
            logger.error(f"Verification audit failed: {e}", exc_info=True)
            return VerificationReport(
                supported=False,
                relevant=False,
                additional_details="Verification system error.",
            )
