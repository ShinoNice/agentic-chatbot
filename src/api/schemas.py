from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.schemas.agent_schemas import GuardrailsReport

# ── Chat ──────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Payload sent by the frontend for every user question."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural-language question.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional client-generated session identifier.",
    )


class VerificationDetail(BaseModel):
    """Mirrors VerificationReport but lives in the API layer."""

    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    relevant: bool = True
    additional_details: str | None = None


class SourceDocument(BaseModel):
    """Simplified reference to a retrieved document chunk."""

    source: str = Field(description="Original file name.")
    page_number: int | None = None
    snippet: str = Field(default="", description="First ~200 chars of the chunk.")


class ChatResponse(BaseModel):
    """Full payload returned for a chat turn."""

    answer: str
    session_id: str | None = None
    relevance_status: str = Field(description="CAN_ANSWER | PARTIAL | NO_MATCH")
    verification: VerificationDetail | None = None
    sources: list[SourceDocument] = Field(default_factory=list)
    iterations: int = 0
    guardrails: GuardrailsReport | None = None


# ── Ingestion ─────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Optional parameters for the ingestion endpoint."""

    namespace: str = Field(
        default="default",
        description="Vector-store namespace / partition.",
    )


class IngestResponse(BaseModel):
    """Summary returned after document ingestion completes."""

    files_processed: list[str]
    total_chunks: int
    status: str = Field(default="completed", pattern="^(pending|completed|failed)$")
    message: str = ""


class UploadResponse(BaseModel):
    """Summary returned after a user uploads a single PDF for per-session querying."""

    filename: str
    total_chunks: int
    namespace: str
    session_id: str


# ── Health ────────────────────────────────────────────────────────────


class LivenessResponse(BaseModel):
    """Liveness probe payload — the process is up and the event loop responsive.

    A 200 here does NOT imply the service is ready to serve traffic; use
    ``/readyz`` for that.
    """

    status: str = "ok"


class ReadinessResponse(BaseModel):
    """Readiness probe payload — the service can handle /chat requests."""

    status: str = Field(default="ok", pattern="^(ok|not_ready)$")
    knowledge_base_ready: bool = False
    vector_store_type: str = "unknown"
    documents_indexed: int | None = None


class HealthResponse(BaseModel):
    """Legacy combined health probe kept for backward compatibility.

    New callers should prefer ``/healthz`` (liveness) and ``/readyz`` (readiness).
    """

    status: str = "ok"
    knowledge_base_ready: bool = False
    vector_store_type: str = "unknown"
    documents_indexed: int | None = None


# ── Audit ────────────────────────────────────────────────────────────


class AuditEventResponse(BaseModel):
    """Audit event as returned by the API."""

    event_id: str
    session_id: str
    timestamp: datetime
    event_type: str
    node_name: str
    details: dict[str, Any]
