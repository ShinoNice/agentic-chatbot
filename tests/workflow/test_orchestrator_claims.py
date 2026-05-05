import pytest
from langchain_core.documents import Document

from src.schemas.claim_schemas import (
    Citation, Claim, ClaimSet, ClaimStatus, DraftedClaim,
)
from src.workflow.orchestrator import RAGOrchestrator


class _Searcher:
    def __init__(self, chunks):
        self.chunks = chunks

    async def aget_retriever(self):
        chunks = self.chunks

        class _R:
            async def ainvoke(_s, q, **kw):
                return chunks

        return _R()


@pytest.mark.asyncio
async def test_orchestrator_claim_path_returns_verified_claims():
    chunks = [
        Document(
            page_content="BM25 is a probabilistic ranking function.",
            metadata={"chunk_id": "ck1", "source": "ir.pdf", "page_number": 3},
        )
    ]
    orch = RAGOrchestrator(_Searcher(chunks))

    class _Drafter:
        async def draft(self, q, ch):
            return ClaimSet(
                claims=[
                    DraftedClaim(
                        id="c1",
                        text="BM25 is a probabilistic ranking function.",
                        citations=[Citation(chunk_id="ck1", quote="probabilistic ranking function")],
                    )
                ]
            )

    class _Verifier:
        async def verify(self, q, claims, chunks):
            for c in claims:
                c.status = ClaimStatus.VERIFIED
            return claims

    class _Relevance:
        async def check(self, q, docs):
            from src.schemas.agent_schemas import RelevanceStatus
            return RelevanceStatus.CAN_ANSWER

    orch.claim_drafter = _Drafter()
    orch.claim_verifier = _Verifier()
    orch.relevance_checker = _Relevance()

    result = await orch.run("what is BM25?", session_id="t")
    assert result["claims"][0].status == ClaimStatus.VERIFIED
    assert "<claim" in result["final_answer"]


@pytest.mark.asyncio
async def test_astream_emits_claim_events():
    chunks = [Document(page_content="x", metadata={"chunk_id": "ck1"})]
    orch = RAGOrchestrator(_Searcher(chunks))

    class _Drafter:
        async def draft(self, q, ch):
            return ClaimSet(
                claims=[
                    DraftedClaim(id="c1", text="x", citations=[Citation(chunk_id="ck1", quote="x")])
                ]
            )

    class _Verifier:
        async def verify(self, q, claims, chunks):
            for c in claims:
                c.status = ClaimStatus.VERIFIED
            return claims

    class _Relevance:
        async def check(self, q, docs):
            from src.schemas.agent_schemas import RelevanceStatus
            return RelevanceStatus.CAN_ANSWER

    orch.claim_drafter = _Drafter()
    orch.claim_verifier = _Verifier()
    orch.relevance_checker = _Relevance()

    events = [evt async for evt in orch.astream_run("q", session_id="t")]
    types = [e[0] for e in events]
    assert "claim_drafted" in types
    assert "claim_verified" in types
    assert "result" in types
    # The result payload should have a string answer and a serialized claims list.
    result_payload = next(e[1] for e in events if e[0] == "result")
    assert isinstance(result_payload.get("answer"), str)
    assert isinstance(result_payload.get("claims"), list)
    assert result_payload["claims"][0]["status"] == "verified"


@pytest.mark.asyncio
async def test_run_truncates_on_budget(monkeypatch):
    from src.core import config_loader
    monkeypatch.setattr(
        config_loader.settings.claim_pipeline, "total_budget_seconds", 0.01, raising=False
    )

    chunks = [Document(page_content="x", metadata={"chunk_id": "ck1"})]
    orch = RAGOrchestrator(_Searcher(chunks))

    import asyncio as _aio

    class _SlowDrafter:
        async def draft(self, q, ch):
            await _aio.sleep(1.0)
            return ClaimSet(claims=[])

    class _Relevance:
        async def check(self, q, docs):
            from src.schemas.agent_schemas import RelevanceStatus
            return RelevanceStatus.CAN_ANSWER

    orch.claim_drafter = _SlowDrafter()
    orch.relevance_checker = _Relevance()

    result = await orch.run("q", session_id="t")
    assert result.get("truncated") is True
