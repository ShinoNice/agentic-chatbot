import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig

from src.engines.base import BaseLLM
from src.schemas.claim_schemas import (
    Citation,
    ClaimSet,
    DraftedClaim,
)
from src.workflow.agents.claim_drafter import ClaimDrafter


class _FakeStructuredRunnable(Runnable):
    """Runnable test double — composes with `prompt | self`."""

    def __init__(self, claim_set: ClaimSet):
        self._claim_set = claim_set
        self.invocations = []

    def invoke(self, input, config: RunnableConfig | None = None, **kwargs):
        self.invocations.append(input)
        return self._claim_set

    async def ainvoke(self, input, config: RunnableConfig | None = None, **kwargs):
        self.invocations.append(input)
        return self._claim_set


class _FakeLLMEngine(BaseLLM):
    def __init__(self, claim_set: ClaimSet):
        self._claim_set = claim_set

    async def generate(self, system_prompt, user_prompt, model_name=None):
        raise NotImplementedError

    def get_model(self):
        outer_set = self._claim_set
        class _M:
            def with_structured_output(_self, schema):
                return _FakeStructuredRunnable(outer_set)
        return _M()


@pytest.mark.asyncio
async def test_drafter_returns_claims_from_chunks():
    chunks = [
        Document(
            page_content="BM25 is a probabilistic ranking function.",
            metadata={"chunk_id": "ck1", "source": "ir.pdf", "page_number": 3},
        )
    ]
    expected = ClaimSet(
        claims=[
            DraftedClaim(
                id="c1",
                text="BM25 is a probabilistic ranking function.",
                citations=[Citation(chunk_id="ck1", quote="probabilistic ranking function")],
            )
        ]
    )
    drafter = ClaimDrafter(_FakeLLMEngine(expected), max_claims=12)
    out = await drafter.draft("what is BM25?", chunks)
    assert isinstance(out, ClaimSet)
    assert len(out.claims) == 1
    assert out.claims[0].citations[0].chunk_id == "ck1"


@pytest.mark.asyncio
async def test_drafter_returns_empty_claimset_when_no_chunks():
    drafter = ClaimDrafter(_FakeLLMEngine(ClaimSet(claims=[])), max_claims=12)
    out = await drafter.draft("what is BM25?", [])
    assert out.claims == []
