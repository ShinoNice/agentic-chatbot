import pytest
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig

from src.engines.base import BaseLLM
from src.schemas.claim_schemas import (
    Citation,
    Claim,
    ClaimStatus,
    ClaimVerificationResult,
)
from src.workflow.agents.claim_verifier import ClaimVerifier


class _FakeStructuredRunnable(Runnable):
    def __init__(self, mapping: dict[str, ClaimVerificationResult]):
        self._mapping = mapping

    def invoke(self, input, config: RunnableConfig | None = None, **kwargs):
        return self._lookup(input)

    async def ainvoke(self, input, config: RunnableConfig | None = None, **kwargs):
        return self._lookup(input)

    def _lookup(self, input):
        text = "".join(getattr(m, "content", "") for m in input.to_messages()) if hasattr(input, "to_messages") else str(input)
        for cid in self._mapping:
            if f"CLAIM ({cid})" in text:
                return self._mapping[cid]
        raise KeyError(f"no canned response found for input: {text[:120]}")


class _FakeEngine(BaseLLM):
    def __init__(self, mapping):
        self._mapping = mapping

    async def generate(self, system_prompt, user_prompt, model_name=None):
        raise NotImplementedError

    def get_model(self):
        m = self._mapping
        class _M:
            def with_structured_output(_s, schema):
                return _FakeStructuredRunnable(m)
        return _M()


@pytest.mark.asyncio
async def test_verifier_marks_each_claim_independently():
    chunks = [
        Document(page_content="BM25 ranks documents.", metadata={"chunk_id": "ck1"}),
        Document(page_content="(unrelated)", metadata={"chunk_id": "ck2"}),
    ]
    claims = [
        Claim(
            id="c1",
            text="BM25 ranks documents.",
            citations=[Citation(chunk_id="ck1", quote="BM25 ranks documents.")],
        ),
        Claim(
            id="c2",
            text="BM25 was invented in 2050.",
            citations=[Citation(chunk_id="ck2", quote="(unrelated)")],
        ),
    ]
    mapping = {
        "c1": ClaimVerificationResult(claim_id="c1", status=ClaimStatus.VERIFIED, note=None),
        "c2": ClaimVerificationResult(
            claim_id="c2", status=ClaimStatus.UNSUPPORTED, note="not in chunk"
        ),
    }
    verifier = ClaimVerifier(_FakeEngine(mapping), concurrency=2)
    out = await verifier.verify(question="what is BM25?", claims=claims, chunks=chunks)
    assert out[0].status == ClaimStatus.VERIFIED
    assert out[1].status == ClaimStatus.UNSUPPORTED
    assert out[1].verifier_note == "not in chunk"


@pytest.mark.asyncio
async def test_verifier_marks_unsupported_when_llm_raises():
    chunks = [Document(page_content="x", metadata={"chunk_id": "ck1"})]
    claims = [
        Claim(id="c1", text="x", citations=[Citation(chunk_id="ck1", quote="x")]),
    ]

    class _Boom(Runnable):
        def invoke(self, input, config: RunnableConfig | None = None, **kwargs):
            raise RuntimeError("simulated LLM failure")

        async def ainvoke(self, input, config: RunnableConfig | None = None, **kwargs):
            raise RuntimeError("simulated LLM failure")

    class _E(BaseLLM):
        async def generate(self, *a, **k):
            raise NotImplementedError

        def get_model(self):
            class _M:
                def with_structured_output(_s, schema):
                    return _Boom()
            return _M()

    verifier = ClaimVerifier(_E(), concurrency=1)
    out = await verifier.verify("q", claims, chunks)
    assert out[0].status == ClaimStatus.UNSUPPORTED
    assert "verification failed" in (out[0].verifier_note or "")
