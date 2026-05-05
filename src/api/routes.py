from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import SystemManager, get_system
from src.api.schemas import (
    AuditEventResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LivenessResponse,
    ReadinessResponse,
    SourceDocument,
    UploadResponse,
    VerificationDetail,
)
from src.core.config_loader import settings
from src.core.logger import logger
from src.mcp.client import get_audit_store
from src.schemas.agent_schemas import RelevanceStatus
from src.schemas.claim_schemas import ClaimStatus

router = APIRouter()

# Hard-coded fallback message returned to the user when retrieval finds no
# relevant context. English-only by design — localization is out of scope for
# the current product surface.
FALLBACK_NO_MATCH_ANSWER = (
    "I couldn't find relevant information in the knowledge base to answer your question."
)


# ── Upload ────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Knowledge Base"],
    summary="Upload a PDF for per-session querying",
)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    system: SystemManager = Depends(get_system),
):
    """Ingest a user-supplied PDF into a session-scoped Pinecone namespace.

    The session's subsequent /chat calls will retrieve only from the uploaded
    document, isolated from the default curated corpus and from other users.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    content = await file.read()
    if len(content) > settings.upload.max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.upload.max_size_mib} MiB limit.",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    # Magic-bytes check: real PDFs always start with "%PDF-".
    if content[:5] != b"%PDF-":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid PDF (missing %PDF- header).",
        )

    try:
        result = await system.upload(content, file.filename, session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Upload failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        )

    return UploadResponse(**result)


# ── Health ────────────────────────────────────────────────────────────


@router.get(
    "/healthz",
    response_model=LivenessResponse,
    tags=["System"],
    summary="Liveness probe — is the process alive?",
)
async def liveness():
    """Cheap, dependency-free probe. Returns 200 whenever the event loop
    is responsive. Do NOT wire this to traffic gating — use ``/readyz``."""
    return LivenessResponse(status="ok")


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    tags=["System"],
    summary="Readiness probe — can this replica serve /chat?",
)
async def readiness(
    response: Response,
    system: SystemManager = Depends(get_system),
):
    """Readiness probe. Returns 503 until the knowledge base is wired up.

    This is the right target for ingress / Container Apps traffic probes —
    a 503 here keeps a starting-up replica out of rotation instead of
    routing user queries to a service that cannot answer them yet.
    """
    if not system.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            knowledge_base_ready=False,
            vector_store_type=system.vector_store_type,
            documents_indexed=None,
        )
    return ReadinessResponse(
        status="ok",
        knowledge_base_ready=True,
        vector_store_type=system.vector_store_type,
        documents_indexed=system.documents_indexed or None,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Legacy combined health probe (prefer /healthz or /readyz)",
    deprecated=True,
)
async def health(system: SystemManager = Depends(get_system)):
    return HealthResponse(
        status="ok",
        knowledge_base_ready=system.is_ready,
        vector_store_type=system.vector_store_type,
        documents_indexed=system.documents_indexed or None,
    )


# ── Ingestion ─────────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Knowledge Base"],
    summary="Ingest PDFs from data/raw/ into the vector store",
)
async def ingest_documents(
    body: IngestRequest = IngestRequest(),
    system: SystemManager = Depends(get_system),
):
    try:
        result = await system.ingest(namespace=body.namespace)
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    if result["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )

    return IngestResponse(**result)


# ── Chat ──────────────────────────────────────────────────────────────


def _build_chat_response(result: dict[str, Any], session_id: str) -> ChatResponse:
    """Translate a raw orchestrator state dict into the wire ChatResponse."""
    answer = result.get("final_answer") or "I couldn't generate an answer."

    rel_status = result.get("relevance_status")
    if isinstance(rel_status, RelevanceStatus):
        relevance_str = rel_status.value
    else:
        relevance_str = str(rel_status) if rel_status else "UNKNOWN"

    if relevance_str == RelevanceStatus.NO_MATCH.value:
        answer = FALLBACK_NO_MATCH_ANSWER

    # Derive a synthetic VerificationDetail from claims so the frontend's
    # existing verification UI keeps working post-cutover.
    raw_claims = result.get("claims") or []
    # Claims may arrive as Pydantic Claim instances (from .run) or as dicts
    # (from astream stream_payload). Normalize to (text, status_str) tuples.
    norm_claims: list[tuple[str, str]] = []
    for c in raw_claims:
        if hasattr(c, "status"):
            status_val = c.status.value if hasattr(c.status, "value") else str(c.status)
            norm_claims.append((getattr(c, "text", ""), status_val))
        elif isinstance(c, dict):
            norm_claims.append((c.get("text", ""), str(c.get("status", ""))))

    verification_detail: VerificationDetail | None = None
    if norm_claims:
        verification_detail = VerificationDetail(
            supported=all(s == ClaimStatus.VERIFIED.value for _, s in norm_claims),
            unsupported_claims=[
                t for t, s in norm_claims if s == ClaimStatus.UNSUPPORTED.value
            ],
            contradictions=[
                t for t, s in norm_claims if s == ClaimStatus.CONTRADICTED.value
            ],
            relevant=True,
            additional_details=None,
        )

    sources: list[SourceDocument] = []
    for doc in result.get("documents", []):
        meta = getattr(doc, "metadata", {}) or {}
        page_content = getattr(doc, "page_content", "")
        sources.append(
            SourceDocument(
                source=meta.get("source", "unknown"),
                page_number=meta.get("page_number") or meta.get("page"),
                snippet=page_content[:200],
            )
        )

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        relevance_status=relevance_str,
        verification=verification_detail,
        sources=sources,
        guardrails=result.get("guardrails_report"),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question against the knowledge base",
)
async def chat(
    body: ChatRequest,
    system: SystemManager = Depends(get_system),
):
    if not system.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not loaded. Call POST /ingest first.",
        )

    session_id = body.session_id or str(uuid.uuid4())

    try:
        result = await system.query(body.question, session_id=session_id)
    except Exception as exc:
        logger.error(f"Query failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the question.",
        )

    return _build_chat_response(result, session_id)


@router.post(
    "/chat/stream",
    tags=["Chat"],
    summary="Stream agent pipeline progress and final answer via SSE",
)
async def chat_stream(
    body: ChatRequest,
    system: SystemManager = Depends(get_system),
):
    if not system.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not loaded. Call POST /ingest first.",
        )

    session_id = body.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            async for event_type, payload in system.query_stream(
                body.question, session_id=session_id
            ):
                if event_type == "result":
                    final = _build_chat_response(payload, session_id)
                    # Yield a dict so the surrounding json.dumps serializes once;
                    # frontend toStreamEvent parses `data` as a JSON object.
                    yield {"event": "result", "data": json.dumps(final.model_dump(mode="json"))}
                else:
                    yield {"event": event_type, "data": json.dumps(payload)}
            yield {"event": "done", "data": "{}"}
        except Exception as exc:
            logger.error(f"Stream failed: {exc}", exc_info=True)
            try:
                err_data = json.dumps({"detail": str(exc)})
            except Exception:
                # Fallback: ensure we always emit a parseable error frame even
                # if the original payload can't be serialized.
                err_data = '{"detail":"stream error (unserializable)"}'
            yield {"event": "error", "data": err_data}

    return EventSourceResponse(event_generator())


# ── Audit Trail ──────────────────────────────────────────────────────


@router.get(
    "/audit/{session_id}",
    response_model=list[AuditEventResponse],
    tags=["Audit"],
    summary="Retrieve audit trail for a session",
)
async def get_audit(session_id: str):
    store = await get_audit_store()
    events = await store.get_trail(session_id)
    if not events:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No audit events found for session {session_id}",
        )
    return [
        AuditEventResponse(
            event_id=e.event_id,
            session_id=e.session_id,
            timestamp=e.timestamp,
            event_type=e.event_type,
            node_name=e.node_name,
            details=e.details,
        )
        for e in events
    ]
