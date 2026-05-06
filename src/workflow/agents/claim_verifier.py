import asyncio

from langchain_core.prompts import ChatPromptTemplate

from src.core.exceptions import AgentError
from src.core.logger import logger
from src.engines.base import BaseLLM
from src.schemas.claim_schemas import (
    Claim,
    ClaimStatus,
    ClaimVerificationResult,
)


_SYSTEM = (
    "You are a strict QA Auditor. For ONE claim, decide if its cited evidence "
    "supports it. Output exactly one of:\n"
    "  - 'verified'      — evidence entails the claim.\n"
    "  - 'unsupported'   — evidence does not entail the claim.\n"
    "  - 'contradicted'  — evidence contradicts the claim.\n"
    "Be pedantic. If the evidence says 'maybe' and the claim says 'always', "
    "that is unsupported."
)


class _Empty:
    page_content = ""


class ClaimVerifier:
    """Per-claim entailment verifier. Runs all pending claims in parallel."""

    def __init__(self, llm_engine: BaseLLM, concurrency: int = 5):
        self.engine = llm_engine
        self.concurrency = max(1, concurrency)
        try:
            self.structured_llm = llm_engine.get_model().with_structured_output(
                ClaimVerificationResult
            )
        except Exception as e:
            raise AgentError(f"LLM does not support structured output: {e}")

    async def verify(
        self, question: str, claims: list[Claim], chunks: list
    ) -> list[Claim]:
        if not claims:
            return []

        chunk_by_id = {(d.metadata or {}).get("chunk_id"): d for d in chunks}
        sem = asyncio.Semaphore(self.concurrency)

        async def _verify_one(claim: Claim) -> Claim:
            async with sem:
                evidence = "\n\n".join(
                    f"[chunk_id={cite.chunk_id}]\n"
                    f"{(chunk_by_id.get(cite.chunk_id) or _Empty()).page_content}"
                    for cite in claim.citations
                )
                if not evidence.strip():
                    claim.status = ClaimStatus.UNSUPPORTED
                    claim.verifier_note = "no evidence cited"
                    return claim
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", _SYSTEM),
                        (
                            "human",
                            "QUESTION: {question}\n\nCLAIM ({claim_id}): {text}\n\n"
                            "EVIDENCE:\n{evidence}\n\n"
                            "Return JSON: claim_id (str), status (str), note (str|null).",
                        ),
                    ]
                )
                try:
                    chain = prompt | self.structured_llm
                    result: ClaimVerificationResult = await chain.ainvoke(
                        {
                            "question": question,
                            "claim_id": claim.id,
                            "text": claim.text,
                            "evidence": evidence,
                        }
                    )
                    claim.status = result.status
                    claim.verifier_note = result.note
                except Exception as e:
                    logger.warning(f"verify({claim.id}) failed: {e}")
                    claim.status = ClaimStatus.UNSUPPORTED
                    claim.verifier_note = f"verification failed: {e}"
                return claim

        return await asyncio.gather(*(_verify_one(c) for c in claims))
