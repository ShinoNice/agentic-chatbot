import pytest
from langchain_core.documents import Document

from src.schemas.claim_schemas import Citation, Claim, ClaimSet, ClaimStatus
from src.workflow.claim_pipeline.repair import repair_claims


class _FakeSearcher:
    def __init__(self, new_chunks):
        self.new_chunks = new_chunks
        self.queries = []

    async def aget_retriever(self):
        outer = self

        class _R:
            async def ainvoke(_self, query, **kw):
                outer.queries.append(query)
                return outer.new_chunks

        return _R()


class _FakeDrafter:
    """Duck-typed drafter: only `draft(question, chunks)` is required."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.calls = []

    async def draft(self, question: str, chunks: list) -> ClaimSet:
        self.calls.append((question, [(d.metadata or {}).get("chunk_id") for d in chunks]))
        return self._mapping[question]


@pytest.mark.asyncio
async def test_repair_only_touches_unsupported_claims_and_increments_attempts():
    verified = Claim(
        id="c1", text="ok", citations=[Citation(chunk_id="ck1", quote="ok")],
        status=ClaimStatus.VERIFIED,
    )
    bad = Claim(
        id="c2", text="bad", citations=[], status=ClaimStatus.UNSUPPORTED,
    )
    new_chunks = [Document(page_content="evidence", metadata={"chunk_id": "ckN"})]
    drafter = _FakeDrafter(
        {
            "bad": ClaimSet(
                claims=[
                    Claim(
                        id="c2",
                        text="bad (rewritten)",
                        citations=[Citation(chunk_id="ckN", quote="evidence")],
                        status=ClaimStatus.PENDING,
                    )
                ]
            )
        }
    )
    searcher = _FakeSearcher(new_chunks)
    out, augmented_chunks = await repair_claims(
        question="q",
        claims=[verified, bad],
        chunks=[Document(page_content="ok", metadata={"chunk_id": "ck1"})],
        drafter=drafter,
        searcher=searcher,
        max_attempts=1,
    )
    assert out[0].status == ClaimStatus.VERIFIED  # untouched
    assert out[1].status == ClaimStatus.PENDING   # repaired, ready for re-verify
    assert out[1].text == "bad (rewritten)"
    assert out[1].attempts == 1
    assert any(d.metadata["chunk_id"] == "ckN" for d in augmented_chunks)


@pytest.mark.asyncio
async def test_repair_skips_when_attempts_at_cap():
    capped = Claim(id="c1", text="x", citations=[], status=ClaimStatus.UNSUPPORTED, attempts=1)
    out, _ = await repair_claims(
        question="q", claims=[capped], chunks=[],
        drafter=_FakeDrafter({}), searcher=_FakeSearcher([]), max_attempts=1,
    )
    assert out[0].attempts == 1
    assert out[0].status == ClaimStatus.UNSUPPORTED


@pytest.mark.asyncio
async def test_repair_marks_unsupported_when_drafter_refuses():
    bad = Claim(id="c1", text="x", citations=[], status=ClaimStatus.UNSUPPORTED)
    drafter = _FakeDrafter({"x": ClaimSet(claims=[])})  # drafter refuses
    searcher = _FakeSearcher([Document(page_content="e", metadata={"chunk_id": "ckX"})])
    out, _ = await repair_claims(
        question="q", claims=[bad], chunks=[],
        drafter=drafter, searcher=searcher, max_attempts=1,
    )
    assert out[0].status == ClaimStatus.UNSUPPORTED
    assert out[0].attempts == 1
    assert "could not ground" in (out[0].verifier_note or "")
