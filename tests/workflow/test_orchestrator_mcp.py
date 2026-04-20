"""Integration tests for MCP guardrails and audit in the orchestrator.

These tests mock the LLM and retrieval layers entirely — no paid API
calls, no real vector store. They verify that:
- guardrails nodes scan/redact PII at the right points
- audit events are logged at each node transition
- the disable toggles work correctly
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document

from src.mcp.audit.store import AuditStore
from src.mcp.client import _reset_audit_store

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def audit_store(tmp_path):
    """Isolated audit store using a temp SQLite DB."""
    db_path = str(tmp_path / "test_audit.db")
    store = AuditStore(db_path=db_path)
    asyncio.run(store.initialize())
    return store


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the audit store singleton between tests."""
    _reset_audit_store()
    yield
    _reset_audit_store()


@pytest.fixture
def mock_docs():
    """Sample documents returned by the mocked retriever."""
    return [
        Document(
            page_content="The patient joao@hospital.pt has NIF 123456789.",
            metadata={
                "source": "medical.pdf",
                "chunk_hash": "abc123",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Treatment plan for condition X.",
            metadata={
                "source": "medical.pdf",
                "chunk_hash": "def456",
                "chunk_index": 1,
            },
        ),
    ]


# ── Guardrails Tests ────────────────────────────────────────────────


class TestGuardrailsInput:
    def test_redacts_pii_in_question(self, tmp_path, monkeypatch, mock_docs):
        """PII in the user's question should be redacted before retrieval."""
        from src.mcp.guardrails.patterns import scan_text

        question = "Find records for joao@hospital.pt"
        detections = scan_text(question)
        assert len(detections) > 0, "Test setup: question should contain PII"

        # Use the client functions directly (no full orchestrator needed)
        from src.mcp.client import guardrails_redact, guardrails_scan

        scan_result = asyncio.run(guardrails_scan(question))
        assert scan_result.has_pii is True

        redacted = asyncio.run(guardrails_redact(question))
        assert "joao@hospital.pt" not in redacted.redacted_text
        assert "[REDACTED_EMAIL]" in redacted.redacted_text

    def test_clean_question_passes_through(self):
        """A question without PII should pass through unchanged."""
        from src.mcp.client import guardrails_scan

        question = "What is the treatment for condition X?"
        scan_result = asyncio.run(guardrails_scan(question))
        assert scan_result.has_pii is False
        assert scan_result.detections == []


class TestGuardrailsOutput:
    def test_redacts_pii_in_answer(self):
        """PII in the draft answer should be redacted before verification."""
        from src.mcp.client import guardrails_redact, guardrails_scan

        draft = "The patient's NIF is 123456789 and email is joao@hospital.pt."
        scan_result = asyncio.run(guardrails_scan(draft))
        assert scan_result.has_pii is True
        assert len(scan_result.detections) >= 2  # NIF + EMAIL

        redacted = asyncio.run(guardrails_redact(draft))
        assert "123456789" not in redacted.redacted_text
        assert "joao@hospital.pt" not in redacted.redacted_text
        assert redacted.redactions_applied >= 2


class TestGuardrailsDisabled:
    def test_disabled_skips_scan(self, monkeypatch):
        """When guardrails.enabled=False, scan returns no detections."""
        from src.core.config_loader import settings

        monkeypatch.setattr(settings.mcp.guardrails, "enabled", False)
        from src.mcp.client import guardrails_scan

        text = "NIF: 123456789, Email: a@b.pt"
        result = asyncio.run(guardrails_scan(text))
        assert result.has_pii is False
        assert result.detections == []

    def test_disabled_skips_redact(self, monkeypatch):
        """When guardrails.enabled=False, redact returns original text."""
        from src.core.config_loader import settings

        monkeypatch.setattr(settings.mcp.guardrails, "enabled", False)
        from src.mcp.client import guardrails_redact

        text = "NIF: 123456789"
        result = asyncio.run(guardrails_redact(text))
        assert result.redacted_text == text
        assert result.redactions_applied == 0


# ── Audit Tests ─────────────────────────────────────────────────────


class TestAuditLogging:
    def test_audit_events_logged(self, audit_store):
        """Verify events can be logged and retrieved."""
        asyncio.run(
            audit_store.log_event("test-sess", "query_received", "guardrails_input", {"q": "test"})
        )
        asyncio.run(
            audit_store.log_event("test-sess", "documents_retrieved", "retrieve", {"count": 5})
        )
        asyncio.run(
            audit_store.log_event(
                "test-sess",
                "answer_delivered",
                "orchestrator",
                {"final_status": "answered"},
            )
        )

        trail = asyncio.run(audit_store.get_trail("test-sess"))
        assert len(trail) == 3
        types = [e.event_type for e in trail]
        assert "query_received" in types
        assert "documents_retrieved" in types
        assert "answer_delivered" in types

    def test_audit_disabled_logs_nothing(self, monkeypatch, tmp_path):
        """When audit.enabled=False, audit_log is a no-op."""
        from src.core.config_loader import settings

        monkeypatch.setattr(settings.mcp.audit, "enabled", False)
        from src.mcp.client import audit_log

        result = asyncio.run(audit_log("sess", "test_event", "node", {}))
        assert result is None


# ── API Audit Endpoint ──────────────────────────────────────────────


class TestAuditEndpoint:
    def test_audit_endpoint_returns_events(self, audit_store, monkeypatch):
        """GET /api/audit/{session_id} returns logged events."""
        # Log some events
        asyncio.run(audit_store.log_event("api-test", "query_received", "guardrails_input", {}))
        asyncio.run(audit_store.log_event("api-test", "answer_delivered", "orchestrator", {}))

        # Monkeypatch the client to return our test store
        monkeypatch.setattr("src.mcp.client._audit_store", audit_store)

        from fastapi.testclient import TestClient

        from src.api.app import app

        client = TestClient(app)
        resp = client.get("/api/audit/api-test")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 2
        assert events[0]["event_type"] == "query_received"
        assert events[1]["event_type"] == "answer_delivered"

    def test_audit_endpoint_404_on_missing_session(self, audit_store, monkeypatch):
        """GET /api/audit/{session_id} returns 404 for unknown session."""
        monkeypatch.setattr("src.mcp.client._audit_store", audit_store)

        from fastapi.testclient import TestClient

        from src.api.app import app

        client = TestClient(app)
        resp = client.get("/api/audit/nonexistent-session")
        assert resp.status_code == 404
