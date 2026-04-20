"""SQLite-backed audit event store.

Stores all workflow events for compliance and traceability.
Uses aiosqlite for async I/O, consistent with the existing
async patterns in the codebase.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from src.schemas.agent_schemas import AuditEvent, AuditSummary

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    node_name TEXT NOT NULL,
    details TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_IDX_SESSION = "CREATE INDEX IF NOT EXISTS idx_session_id ON audit_events(session_id)"
_CREATE_IDX_TYPE = "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)"

_INSERT = """
INSERT INTO audit_events (event_id, session_id, timestamp, event_type, node_name, details)
VALUES (?, ?, ?, ?, ?, ?)
"""

_SELECT_BY_SESSION = """
SELECT event_id, session_id, timestamp, event_type, node_name, details
FROM audit_events
WHERE session_id = ?
ORDER BY timestamp ASC
"""

_DELETE_OLD = """
DELETE FROM audit_events WHERE timestamp < ?
"""


class AuditStore:
    """Async SQLite-backed audit event persistence."""

    def __init__(self, db_path: str = "data/audit/audit.db"):
        self.db_path = db_path

    async def initialize(self) -> None:
        """Create tables and indices if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(_CREATE_TABLE)
            await db.execute(_CREATE_IDX_SESSION)
            await db.execute(_CREATE_IDX_TYPE)
            await db.commit()

    async def log_event(
        self,
        session_id: str,
        event_type: str,
        node_name: str,
        details: dict[str, Any],
    ) -> str:
        """Log a single audit event. Returns the generated event_id."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                _INSERT,
                (
                    event_id,
                    session_id,
                    timestamp,
                    event_type,
                    node_name,
                    json.dumps(details),
                ),
            )
            await db.commit()
        return event_id

    async def get_trail(self, session_id: str) -> list[AuditEvent]:
        """Retrieve all events for a session, ordered by timestamp."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(_SELECT_BY_SESSION, (session_id,))
            rows = await cursor.fetchall()
        return [
            AuditEvent(
                event_id=row[0],
                session_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                event_type=row[3],
                node_name=row[4],
                details=json.loads(row[5]),
            )
            for row in rows
        ]

    async def get_summary(self, session_id: str) -> AuditSummary:
        """Build an aggregated summary from a session's events."""
        events = await self.get_trail(session_id)
        if not events:
            return AuditSummary(
                session_id=session_id,
                total_events=0,
                documents_accessed=0,
                pii_detections=0,
                pii_redactions=0,
                workflow_duration_ms=0,
                final_status="unknown",
            )

        docs_accessed = sum(1 for e in events if e.event_type == "documents_retrieved")
        pii_detections = sum(
            e.details.get("count", 0) for e in events if e.event_type == "pii_detected"
        )
        pii_redactions = sum(
            e.details.get("redactions", 0) for e in events if e.event_type == "pii_redacted"
        )

        first_ts = events[0].timestamp
        last_ts = events[-1].timestamp
        duration_ms = int((last_ts - first_ts).total_seconds() * 1000)

        # Final status from the last answer_delivered event, or "unknown"
        delivered = [e for e in events if e.event_type == "answer_delivered"]
        final_status = (
            delivered[-1].details.get("final_status", "unknown") if delivered else "unknown"
        )

        return AuditSummary(
            session_id=session_id,
            total_events=len(events),
            documents_accessed=docs_accessed,
            pii_detections=pii_detections,
            pii_redactions=pii_redactions,
            workflow_duration_ms=duration_ms,
            final_status=final_status,
        )

    async def cleanup(self, retention_days: int) -> int:
        """Delete events older than retention_days. Returns count deleted."""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(_DELETE_OLD, (cutoff,))
            deleted = cursor.rowcount
            await db.commit()
        return deleted
