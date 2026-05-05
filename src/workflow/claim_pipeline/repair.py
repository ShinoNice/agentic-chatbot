import asyncio
from typing import Tuple

from langchain_core.documents import Document

from src.core.logger import logger
from src.schemas.claim_schemas import Claim, ClaimStatus


async def _retrieve_for_claim(searcher, claim: Claim) -> list[Document]:
    retriever = await searcher.aget_retriever()
    docs = await retriever.ainvoke(claim.text)
    return docs or []


async def _redraft_one(drafter, claim: Claim, chunks: list[Document]) -> Claim:
    """Ask the drafter for a single-claim ClaimSet scoped to this claim's text."""
    claim_set = await drafter.draft(claim.text, chunks)
    pending = claim_set.to_pending_claims()
    if not pending:
        claim.status = ClaimStatus.UNSUPPORTED
        claim.verifier_note = "repair: drafter could not ground claim"
        return claim
    repaired = pending[0]
    repaired.id = claim.id  # preserve original id
    repaired.attempts = claim.attempts  # caller increments
    return repaired


async def repair_claims(
    *,
    question: str,
    claims: list[Claim],
    chunks: list[Document],
    drafter,
    searcher,
    max_attempts: int = 1,
) -> Tuple[list[Claim], list[Document]]:
    """Per-claim re-retrieve + redraft. Returns (updated_claims, augmented_chunks).

    Verified claims are left untouched. Unsupported/contradicted claims with
    attempts < max_attempts are repaired in parallel.
    """

    to_repair = [
        c for c in claims
        if c.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED)
        and c.attempts < max_attempts
    ]
    if not to_repair:
        return claims, chunks

    logger.info(f"repair_claims: repairing {len(to_repair)} claim(s)")

    async def _one(claim: Claim):
        new_chunks = await _retrieve_for_claim(searcher, claim)
        repaired = await _redraft_one(drafter, claim, new_chunks)
        repaired.attempts = claim.attempts + 1
        return repaired, new_chunks

    results = await asyncio.gather(*(_one(c) for c in to_repair))

    repaired_by_id = {r.id: r for r, _ in results}
    augmented = list(chunks)
    seen_ids = {(d.metadata or {}).get("chunk_id") for d in augmented}
    for _, new_docs in results:
        for d in new_docs:
            cid = (d.metadata or {}).get("chunk_id")
            if cid and cid not in seen_ids:
                augmented.append(d)
                seen_ids.add(cid)

    updated = [repaired_by_id.get(c.id, c) for c in claims]
    return updated, augmented
