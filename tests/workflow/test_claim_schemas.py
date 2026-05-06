import pytest
from pydantic import ValidationError

from src.schemas.claim_schemas import (
    Citation,
    Claim,
    ClaimSet,
    ClaimStatus,
    ClaimVerificationResult,
)


def test_citation_requires_chunk_id_and_quote():
    c = Citation(chunk_id="abc123", quote="hello world")
    assert c.chunk_id == "abc123"
    assert c.quote == "hello world"
    with pytest.raises(ValidationError):
        Citation(chunk_id="abc123")


def test_claim_default_status_is_pending_and_attempts_zero():
    claim = Claim(id="c1", text="x", citations=[Citation(chunk_id="a", quote="q")])
    assert claim.status == ClaimStatus.PENDING
    assert claim.attempts == 0
    assert claim.verifier_note is None


def test_claim_set_caps_claims_via_validator():
    too_many = [
        Claim(id=f"c{i}", text=str(i), citations=[Citation(chunk_id="a", quote="q")])
        for i in range(20)
    ]
    with pytest.raises(ValidationError):
        ClaimSet(claims=too_many, max_claims=12)


def test_verification_result_round_trip():
    r = ClaimVerificationResult(
        claim_id="c1", status=ClaimStatus.VERIFIED, note=None
    )
    assert r.model_dump()["status"] == "verified"
