from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RelevanceStatus(str, Enum):
    CAN_ANSWER = "CAN_ANSWER"
    PARTIAL = "PARTIAL"
    NO_MATCH = "NO_MATCH"


class RelevanceResponse(BaseModel):
    """Structured output for the RelevanceChecker agent."""

    status: RelevanceStatus = Field(
        description="How well the context addresses the question.",
    )
    reasoning: str = Field(
        description="Brief justification for the assigned status.",
    )


class VerificationReport(BaseModel):
    """Structured output for the VerificationAgent."""

    supported: bool = Field(
        description="True if the answer is fully supported by the context.",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims in the answer with no evidence in the context.",
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="Statements that directly conflict with the context.",
    )
    relevant: bool = Field(
        description="True if the answer addresses the user's question.",
    )
    additional_details: Optional[str] = Field(
        default=None,
        description="Extra nuance or context found during the audit.",
    )


class ResearchOutput(BaseModel):
    """Standardized output for the ResearchAgent generation step."""

    draft_answer: str = Field(description="Generated answer based on documents.")
    context_used: str = Field(description="Raw text used to generate the answer.")
