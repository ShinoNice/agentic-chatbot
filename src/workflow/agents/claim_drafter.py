from langchain_core.prompts import ChatPromptTemplate

from src.core.exceptions import AgentError, LLMResponseError
from src.core.logger import logger
from src.engines.base import BaseLLM
from src.schemas.claim_schemas import ClaimSet


_SYSTEM = (
    "You are a precise Research Assistant. You will produce a structured answer "
    "as a list of atomic, individually-cited factual claims.\n\n"
    "RULES:\n"
    "1. Each claim is ONE factual statement. No conjunctions, no compound facts.\n"
    "2. Every claim MUST cite at least one chunk with a verbatim QUOTE from that chunk.\n"
    "3. Use a stable id per claim: c1, c2, c3, …\n"
    "4. If you cannot ground a claim in the chunks, OMIT it. Do NOT fabricate.\n"
    "5. Return AT MOST {max_claims} claims."
)


def _format_chunks(chunks: list) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(
        f"[chunk_id={d.metadata.get('chunk_id', '?')} source={d.metadata.get('source', '?')}]\n"
        f"{d.page_content}"
        for d in chunks
    )


class ClaimDrafter:
    """Drafts the answer as a structured ClaimSet using LLM JSON-mode."""

    def __init__(self, llm_engine: BaseLLM, max_claims: int = 12):
        self.engine = llm_engine
        self.max_claims = max_claims
        try:
            self.structured_llm = llm_engine.get_model().with_structured_output(ClaimSet)
        except Exception as e:
            raise AgentError(f"LLM does not support structured output: {e}")

    async def draft(self, question: str, chunks: list) -> ClaimSet:
        if not chunks:
            logger.info("ClaimDrafter: no chunks; returning empty ClaimSet.")
            return ClaimSet(claims=[])

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM.format(max_claims=self.max_claims)),
                (
                    "human",
                    "QUESTION: {question}\n\nCHUNKS:\n{chunks}\n\n"
                    "Return a JSON object matching the ClaimSet schema.",
                ),
            ]
        )
        try:
            chain = prompt | self.structured_llm
            result: ClaimSet = await chain.ainvoke(
                {"question": question, "chunks": _format_chunks(chunks)}
            )
            logger.info(f"ClaimDrafter: produced {len(result.claims)} claim(s)")
            return result
        except Exception as e:
            raise LLMResponseError(f"ClaimDrafter failed: {e}")
