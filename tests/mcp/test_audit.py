"""Tests for the audit trail store.

All tests use tmp_path for isolated SQLite DBs — no shared state.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from src.mcp.audit.store import AuditStore


@pytest.fixture
def store(tmp_path):
    """Create an AuditStore with a temp DB and initialize it."""
    db_path = str(tmp_path / "test_audit.db")
    s = AuditStore(db_path=db_path)
    asyncio.run(s.initialize())
    return s


class TestLogEvent:
    def test_log_event_returns_uuid(self, store):
        event_id = asyncio.run(
            store.log_event("sess-1", "query_received", "guardrails_input", {"q": "test"})
        )
        assert event_id is not None
        assert len(event_id) == 36  # UUID4 format

    def test_log_multiple_events(self, store):
        id1 = asyncio.run(store.log_event("sess-1", "query_received", "guardrails_input", {}))
        id2 = asyncio.run(
            store.log_event("sess-1", "documents_retrieved", "retrieve", {"count": 10})
        )
        assert id1 != id2


class TestGetTrail:
    def test_get_trail_returns_events_in_order(self, store):
        asyncio.run(store.log_event("sess-1", "query_received", "guardrails_input", {}))
        asyncio.run(store.log_event("sess-1", "documents_retrieved", "retrieve", {}))
        asyncio.run(store.log_event("sess-1", "answer_generated", "research", {}))

        trail = asyncio.run(store.get_trail("sess-1"))
        assert len(trail) == 3
        assert trail[0].event_type == "query_received"
        assert trail[1].event_type == "documents_retrieved"
        assert trail[2].event_type == "answer_generated"
        # Timestamps should be non-decreasing
        assert trail[0].timestamp <= trail[1].timestamp <= trail[2].timestamp

    def test_get_trail_empty_session(self, store):
        trail = asyncio.run(store.get_trail("nonexistent"))
        assert trail == []

    def test_get_trail_isolates_sessions(self, store):
        asyncio.run(store.log_event("sess-A", "query_received", "node", {}))
        asyncio.run(store.log_event("sess-B", "query_received", "node", {}))

        trail_a = asyncio.run(store.get_trail("sess-A"))
        trail_b = asyncio.run(store.get_trail("sess-B"))
        assert len(trail_a) == 1
        assert len(trail_b) == 1
        assert trail_a[0].session_id == "sess-A"
        assert trail_b[0].session_id == "sess-B"


class TestGetSummary:
    def test_get_summary_counts(self, store):
        asyncio.run(store.log_event("sess-1", "query_received", "guardrails_input", {}))
        asyncio.run(store.log_event("sess-1", "documents_retrieved", "retrieve", {"count": 30}))
        asyncio.run(
            store.log_event(
                "sess-1",
                "answer_delivered",
                "orchestrator",
                {"final_status": "answered"},
            )
        )

        summary = asyncio.run(store.get_summary("sess-1"))
        assert summary.session_id == "sess-1"
        assert summary.total_events == 3
        assert summary.documents_accessed == 1
        assert summary.final_status == "answered"

    def test_get_summary_pii_counts(self, store):
        asyncio.run(
            store.log_event(
                "sess-1",
                "pii_detected",
                "guardrails_input",
                {"count": 2, "pii_types": ["EMAIL", "NIF"]},
            )
        )
        asyncio.run(
            store.log_event(
                "sess-1",
                "pii_redacted",
                "guardrails_output",
                {"redactions": 3, "strategy": "mask"},
            )
        )

        summary = asyncio.run(store.get_summary("sess-1"))
        assert summary.pii_detections == 2
        assert summary.pii_redactions == 3

    def test_get_summary_empty_session(self, store):
        summary = asyncio.run(store.get_summary("nonexistent"))
        assert summary.total_events == 0
        assert summary.final_status == "unknown"


class TestCleanup:
    def test_cleanup_removes_old_events(self, store):
        # Log an event, then backdate it via direct SQL
        asyncio.run(store.log_event("old-sess", "query_received", "node", {}))

        import aiosqlite

        async def backdate():
            old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat()
            async with aiosqlite.connect(store.db_path) as db:
                await db.execute(
                    "UPDATE audit_events SET timestamp = ? WHERE session_id = ?",
                    (old_ts, "old-sess"),
                )
                await db.commit()

        asyncio.run(backdate())

        deleted = asyncio.run(store.cleanup(retention_days=90))
        assert deleted == 1

        trail = asyncio.run(store.get_trail("old-sess"))
        assert len(trail) == 0

    def test_cleanup_keeps_recent_events(self, store):
        asyncio.run(store.log_event("new-sess", "query_received", "node", {}))

        deleted = asyncio.run(store.cleanup(retention_days=90))
        assert deleted == 0

        trail = asyncio.run(store.get_trail("new-sess"))
        assert len(trail) == 1


class TestInitialize:
    def test_initialize_creates_directory(self, tmp_path):
        nested_path = str(tmp_path / "deep" / "nested" / "audit.db")
        s = AuditStore(db_path=nested_path)
        asyncio.run(s.initialize())
        # Should not raise, and DB should be usable
        event_id = asyncio.run(s.log_event("sess-1", "query_received", "node", {}))
        assert event_id is not None
