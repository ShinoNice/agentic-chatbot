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

_SYSTEM_REPAIR = (
    "You are repairing a SINGLE prior claim. The user message contains the "
    "original claim text and a small set of evidence chunks just retrieved "
    "specifically for it.\n\n"
    "RULES:\n"
    "1. Return EXACTLY ONE claim restating the original idea in a form the "
    "evidence supports — OR return zero claims if the evidence does not "
    "support it. Do NOT pad with related claims.\n"
    "2. Use id 'c1'.\n"
    "3. Cite at least one chunk with a verbatim QUOTE from that chunk.\n"
    "4. Do NOT fabricate. If no chunk supports the claim, return an empty list."
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
    """Drafts the answer as a structured ClaimSet using LLM JSON-mode.

    Two modes:
      - default (single_claim_mode=False): produces up to ``max_claims`` claims
        for an end-user question; used by ``node_draft_claims``.
      - single_claim_mode=True: produces exactly 0 or 1 claim, used by repair
        to restate a specific prior claim against fresh evidence — much shorter
        prompt and shorter output → ~5–10s vs ~25-30s of the default mode.
    """

    def __init__(
        self,
        llm_engine: BaseLLM,
        max_claims: int = 12,
        single_claim_mode: bool = False,
    ):
        self.engine = llm_engine
        self.max_claims = 1 if single_claim_mode else max_claims
        self.single_claim_mode = single_claim_mode
        try:
            self.structured_llm = llm_engine.get_model().with_structured_output(ClaimSet)
        except Exception as e:
            raise AgentError(f"LLM does not support structured output: {e}")

    async def draft(self, question: str, chunks: list) -> ClaimSet:
        if not chunks:
            logger.info("ClaimDrafter: no chunks; returning empty ClaimSet.")
            return ClaimSet(claims=[])

        if self.single_claim_mode:
            system_msg = _SYSTEM_REPAIR
            human_msg = (
                "ORIGINAL CLAIM: {question}\n\nEVIDENCE CHUNKS:\n{chunks}\n\n"
                "Return a JSON object matching the ClaimSet schema with EXACTLY "
                "0 or 1 claims."
            )
        else:
            system_msg = _SYSTEM.format(max_claims=self.max_claims)
            human_msg = (
                "QUESTION: {question}\n\nCHUNKS:\n{chunks}\n\n"
                "Return a JSON object matching the ClaimSet schema."
            )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_msg), ("human", human_msg)]
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
