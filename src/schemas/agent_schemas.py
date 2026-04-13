from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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


# ── MCP Guardrails Schemas ───────────────────────────────────────────


class PIIDetection(BaseModel):
    """A single PII detection within scanned text."""

    pii_type: str = Field(
        description="Type of PII detected (e.g. NIF, EMAIL, PHONE_PT, CREDIT_CARD).",
    )
    value: str = Field(description="The matched text.")
    start: int = Field(description="Character offset of match start.")
    end: int = Field(description="Character offset of match end.")
    confidence: float = Field(
        default=1.0,
        description="Detection confidence (1.0 for regex, <1.0 for heuristic).",
    )


class PIIScanResult(BaseModel):
    """Result of scanning text for PII patterns."""

    has_pii: bool = Field(description="Whether any PII was detected.")
    detections: List[PIIDetection] = Field(default_factory=list)
    scanned_length: int = Field(description="Length of the scanned text.")


class RedactedResult(BaseModel):
    """Result of redacting PII from text."""

    original_length: int
    redacted_text: str
    redactions_applied: int
    strategy: str


class GuardrailsReport(BaseModel):
    """Summary of PII scanning and redaction across input and output."""

    input_pii_found: bool = False
    input_redactions: int = 0
    output_pii_found: bool = False
    output_redactions: int = 0
    detections: List[PIIDetection] = Field(default_factory=list)


# ── MCP Audit Trail Schemas ─────────────────────────────────────────


class AuditEvent(BaseModel):
    """A single audit trail event logged during workflow execution."""

    event_id: str = Field(description="Unique event identifier (UUID4).")
    session_id: str = Field(description="Chat session this event belongs to.")
    timestamp: datetime = Field(description="UTC timestamp of the event.")
    event_type: str = Field(
        description="Event category (e.g. query_received, pii_detected).",
    )
    node_name: str = Field(description="LangGraph node that produced this event.")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific payload.",
    )


class AuditSummary(BaseModel):
    """Aggregated summary of a session's audit trail."""

    session_id: str
    total_events: int
    documents_accessed: int
    pii_detections: int
    pii_redactions: int
    workflow_duration_ms: int
    final_status: str = Field(
        description="Final workflow outcome (answered, no_match, error).",
    )
