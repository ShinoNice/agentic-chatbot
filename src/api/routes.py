"""
API route definitions for the Agentic RAG Chatbot.

Three endpoints:
    GET  /health  – liveness / readiness probe
    POST /ingest  – trigger PDF ingestion pipeline
    POST /chat    – ask a question against the knowledge base
"""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import SystemManager, get_system
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceDocument,
    VerificationDetail,
)
from src.core.logger import logger
from src.schemas.agent_schemas import RelevanceStatus

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health / readiness check",
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
    # Guard – knowledge base must be loaded first
    if not system.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not loaded. Call POST /ingest first.",
        )

    session_id = body.session_id or str(uuid.uuid4())

    try:
        result = await system.query(body.question)
    except Exception as exc:
        logger.error(f"Query failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the question: {exc}",
        )

    # ── Map internal AgentState → API response ────────────────────────

    # Answer
    answer = result.get("draft_answer") or "I couldn't generate an answer."

    # Relevance
    rel_status = result.get("relevance_status")
    if isinstance(rel_status, RelevanceStatus):
        relevance_str = rel_status.value
    else:
        relevance_str = str(rel_status) if rel_status else "UNKNOWN"

    if relevance_str == RelevanceStatus.NO_MATCH.value:
        answer = (
            "I couldn't find relevant information in the knowledge base "
            "to answer your question."
        )

    # Verification
    verification_detail = None
    verification = result.get("verification")
    if verification is not None:
        verification_detail = VerificationDetail(
            supported=verification.supported,
            unsupported_claims=verification.unsupported_claims,
            contradictions=verification.contradictions,
            relevant=verification.relevant,
            additional_details=verification.additional_details,
        )

    # Source documents (lightweight references)
    sources: List[SourceDocument] = []
    for doc in result.get("documents", []):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        sources.append(
            SourceDocument(
                source=meta.get("source", "unknown"),
                page_number=meta.get("page_number") or meta.get("page"),
                snippet=doc.page_content[:200] if hasattr(doc, "page_content") else "",
            )
        )

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        relevance_status=relevance_str,
        verification=verification_detail,
        sources=sources,
        iterations=result.get("iterations", 0),
    )
