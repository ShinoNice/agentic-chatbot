"""Tests for the PDF upload endpoint and session-scoped orchestrator routing.

These tests mock the heavy ML dependencies (DocumentProcessor, Pinecone,
HybridSearcher, RAGOrchestrator) so they run fast and need no network or
vendor credentials.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api import dependencies as deps_module
from src.api.app import create_app


@pytest.fixture
def fake_pdf_bytes() -> bytes:
    """A minimal valid PDF byte stream — just a header plus EOF marker.

    Real parsing is mocked, so this only needs to pass the ``.pdf`` extension
    check and the size gate.
    """
    return b"%PDF-1.4\n%EOF\n"


@pytest.fixture
def patched_system(monkeypatch):
    """Install a SystemManager whose heavy pieces are MagicMocks.

    Returns the SystemManager instance so tests can assert on calls.
    """
    system = deps_module.SystemManager.__new__(deps_module.SystemManager)
    system.processor = MagicMock()
    system.vector_manager = MagicMock()
    system.searcher = MagicMock()
    system.orchestrator = MagicMock()
    system._files_indexed = []
    system._total_chunks = 0
    system._session_orchestrators = {}
    system._session_searchers = {}

    # Default corpus is "ready" so the /chat guard passes.
    type(system).is_ready = property(lambda self: True)

    # Mock processor.process() to return 2 small chunks regardless of input.
    chunks = [
        Document(
            page_content="chunk 1 from uploaded pdf",
            metadata={"source": "user.pdf", "chunk_hash": "h1"},
        ),
        Document(
            page_content="chunk 2 from uploaded pdf",
            metadata={"source": "user.pdf", "chunk_hash": "h2"},
        ),
    ]
    system.processor.process.return_value = chunks

    # create_index returns a dummy vector store (unused by tests).
    system.vector_manager.create_index.return_value = MagicMock()

    # Install as the module-level singleton.
    monkeypatch.setattr(deps_module, "_system", system)
    return system


@pytest.fixture
def client(patched_system):
    app = create_app()
    return TestClient(app)


def test_upload_accepts_pdf_and_returns_session_namespace(
    client, fake_pdf_bytes, patched_system
):
    """Happy path: POST /api/upload with a PDF returns chunk count + namespace."""
    response = client.post(
        "/api/upload",
        files={"file": ("mydoc.pdf", fake_pdf_bytes, "application/pdf")},
        data={"session_id": "sess-123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "mydoc.pdf"
    assert body["total_chunks"] == 2
    assert body["namespace"] == "session-sess-123"
    assert body["session_id"] == "sess-123"

    # Verify the upload was routed into the session-scoped namespace.
    patched_system.vector_manager.create_index.assert_called_once()
    _, kwargs = patched_system.vector_manager.create_index.call_args
    assert kwargs["namespace"] == "session-sess-123"


def test_upload_rejects_non_pdf(client):
    """Non-PDF extensions are rejected with 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        data={"session_id": "sess-1"},
    )
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_upload_rejects_oversize_file(client):
    """Files over 20 MB are rejected with 413."""
    big_pdf = b"%PDF-1.4\n" + b"x" * (20 * 1024 * 1024 + 1)
    response = client.post(
        "/api/upload",
        files={"file": ("huge.pdf", big_pdf, "application/pdf")},
        data={"session_id": "sess-1"},
    )
    assert response.status_code == 413


def test_upload_installs_session_orchestrator(client, fake_pdf_bytes, patched_system):
    """After upload, a per-session orchestrator is stored so /chat can route to it."""
    response = client.post(
        "/api/upload",
        files={"file": ("user.pdf", fake_pdf_bytes, "application/pdf")},
        data={"session_id": "sess-xyz"},
    )
    assert response.status_code == 200
    assert "sess-xyz" in patched_system._session_orchestrators


def test_chat_uses_session_orchestrator_when_present(client, patched_system):
    """If a session has uploaded a doc, /chat routes through that session's orchestrator."""
    # Pre-install a session orchestrator for sess-route with a distinguishable result.
    session_orch = MagicMock()
    session_orch.run = AsyncMock(
        return_value={
            "draft_answer": "from session orchestrator",
            "relevance_status": "CAN_ANSWER",
            "documents": [],
            "iterations": 1,
        }
    )
    patched_system._session_orchestrators["sess-route"] = session_orch

    # Default orchestrator returns something different so we can tell them apart.
    patched_system.orchestrator.run = AsyncMock(
        return_value={
            "draft_answer": "from default orchestrator",
            "relevance_status": "CAN_ANSWER",
            "documents": [],
            "iterations": 1,
        }
    )

    response = client.post(
        "/api/chat",
        json={"question": "anything", "session_id": "sess-route"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "from session orchestrator"
    session_orch.run.assert_awaited_once()
    patched_system.orchestrator.run.assert_not_called()


def test_chat_falls_back_to_default_without_session_upload(client, patched_system):
    """Chat without an uploaded-doc session uses the default curated orchestrator."""
    patched_system.orchestrator.run = AsyncMock(
        return_value={
            "draft_answer": "from default orchestrator",
            "relevance_status": "CAN_ANSWER",
            "documents": [],
            "iterations": 1,
        }
    )

    response = client.post(
        "/api/chat",
        json={"question": "anything", "session_id": "no-upload"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "from default orchestrator"
    patched_system.orchestrator.run.assert_awaited_once()
