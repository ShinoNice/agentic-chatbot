"""In-process MCP client wrapper for LangGraph nodes.

Provides simple async functions that LangGraph nodes import directly.
Each function respects the enabled/disabled toggle from settings —
when disabled, calls are no-ops returning safe defaults.
"""

from __future__ import annotations

from src.mcp.audit.store import AuditStore
from src.mcp.guardrails.patterns import scan_text
from src.schemas.agent_schemas import PIIScanResult, RedactedResult

_audit_store: AuditStore | None = None


async def get_audit_store() -> AuditStore:
    """Lazy-initialise the audit store singleton."""
    global _audit_store
    if _audit_store is None:
        from src.core.config_loader import settings

        _audit_store = AuditStore(settings.mcp.audit.db_path)
        await _audit_store.initialize()
    return _audit_store


def _reset_audit_store() -> None:
    """Reset the audit store singleton (for tests)."""
    global _audit_store
    _audit_store = None


async def guardrails_scan(text: str) -> PIIScanResult:
    """Scan text for PII. Returns empty result when guardrails disabled."""
    from src.core.config_loader import settings

    if not settings.mcp.guardrails.enabled:
        return PIIScanResult(has_pii=False, detections=[], scanned_length=len(text))

    detections = scan_text(text)
    return PIIScanResult(
        has_pii=len(detections) > 0,
        detections=detections,
        scanned_length=len(text),
    )


async def guardrails_redact(text: str) -> RedactedResult:
    """Redact PII from text using configured strategy.

    Returns the original text unchanged when guardrails disabled.
    """
    import hashlib

    from src.core.config_loader import settings

    if not settings.mcp.guardrails.enabled:
        return RedactedResult(
            original_length=len(text),
            redacted_text=text,
            redactions_applied=0,
            strategy="none",
        )

    strategy = settings.mcp.guardrails.redaction_strategy
    detections = scan_text(text)

    redacted = text
    for det in sorted(detections, key=lambda d: d.start, reverse=True):
        if strategy == "mask":
            replacement = f"[REDACTED_{det.pii_type}]"
        elif strategy == "hash":
            h = hashlib.sha256(det.value.encode()).hexdigest()[:8]
            replacement = f"[{det.pii_type}:{h}]"
        elif strategy == "remove":
            replacement = ""
        else:
            replacement = f"[REDACTED_{det.pii_type}]"
        redacted = redacted[: det.start] + replacement + redacted[det.end :]

    return RedactedResult(
        original_length=len(text),
        redacted_text=redacted,
        redactions_applied=len(detections),
        strategy=strategy,
    )


async def audit_log(
    session_id: str,
    event_type: str,
    node_name: str,
    details: dict,
) -> str | None:
    """Log an audit event. No-op when audit disabled. Returns event_id or None."""
    from src.core.config_loader import settings

    if not settings.mcp.audit.enabled:
        return None

    store = await get_audit_store()
    return await store.log_event(session_id, event_type, node_name, details)
