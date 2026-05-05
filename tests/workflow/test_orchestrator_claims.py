import pytest
from langchain_core.documents import Document

from src.schemas.claim_schemas import (
    Citation, Claim, ClaimSet, ClaimStatus,
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
async def test_orchestrator_claim_path_returns_verified_claims(monkeypatch):
    from src.core import config_loader
    monkeypatch.setattr(
        config_loader.settings.claim_pipeline, "enabled", True, raising=False
    )

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
                    Claim(
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
async def test_orchestrator_legacy_path_still_works_when_flag_off():
    """Sanity: with the flag off, the existing graph path still completes."""
    chunks = [Document(page_content="x", metadata={"chunk_id": "ck1"})]
    orch = RAGOrchestrator(_Searcher(chunks))
    # Don't enable the flag. We don't invoke .run() (it'd hit a real LLM); we
    # just assert the build succeeded with the legacy segment.
    nodes = orch.app.get_graph().nodes
    assert "research" in nodes
    assert "verify" in nodes
