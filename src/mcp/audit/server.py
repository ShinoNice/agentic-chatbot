"""Audit Trail MCP Server — event logging and retrieval tools.

Built with FastMCP. Can be used in-process (imported directly) or
run standalone via ``python -m src.mcp.audit``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src.mcp.audit.store import AuditStore

mcp_server = FastMCP("audit")

_store: AuditStore | None = None


async def get_store() -> AuditStore:
    """Lazy-initialise the audit store singleton."""
    global _store
    if _store is None:
        from src.core.config_loader import settings

        _store = AuditStore(settings.mcp.audit.db_path)
        await _store.initialize()
    return _store


def _reset_store() -> None:
    """Reset the store singleton (for tests)."""
    global _store
    _store = None


@mcp_server.tool()
async def log_event(
    session_id: str,
    event_type: str,
    node_name: str,
    details: Dict[str, Any],
) -> str:
    """Log a single audit event. Returns the event ID."""
    store = await get_store()
    return await store.log_event(session_id, event_type, node_name, details)


@mcp_server.tool()
async def get_audit_trail(session_id: str) -> List[dict]:
    """Retrieve all events for a session, ordered by timestamp."""
    store = await get_store()
    events = await store.get_trail(session_id)
    return [e.model_dump(mode="json") for e in events]


@mcp_server.tool()
async def get_audit_summary(session_id: str) -> dict:
    """Get an aggregated summary of a session's audit trail."""
    store = await get_store()
    summary = await store.get_summary(session_id)
    return summary.model_dump()
