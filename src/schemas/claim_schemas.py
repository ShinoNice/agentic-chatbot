from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ClaimStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class Citation(BaseModel):
    chunk_id: str = Field(description="Stable id of the source chunk.")
    quote: str = Field(description="Verbatim span from the chunk supporting the claim.")


class Claim(BaseModel):
    id: str = Field(description="Stable id within this answer, e.g. 'c1'.")
    text: str = Field(description="One atomic factual statement, plain prose.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Evidence supporting the claim. May be empty if status is unsupported.",
    )
    status: ClaimStatus = Field(default=ClaimStatus.PENDING)
    verifier_note: str | None = Field(default=None)
    attempts: int = Field(default=0, ge=0, description="Repair attempts on this claim.")


class ClaimSet(BaseModel):
    """Drafter structured output: a list of claims under a hard cap."""

    claims: list[Claim] = Field(default_factory=list)
    max_claims: int = Field(default=12, exclude=True)

    @model_validator(mode="after")
    def _enforce_cap(self) -> "ClaimSet":
        if len(self.claims) > self.max_claims:
            raise ValueError(
                f"Drafter returned {len(self.claims)} claims, exceeds cap {self.max_claims}."
            )
        return self


class ClaimVerificationResult(BaseModel):
    """Per-claim entailment outcome from the verifier."""

    claim_id: str
    status: ClaimStatus
    note: str | None = None
