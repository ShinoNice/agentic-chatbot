"""Guardrails MCP Server — PII scanning and redaction tools.

Built with FastMCP. Can be used in-process (imported directly) or
run standalone via ``python -m src.mcp.guardrails``.
"""

from __future__ import annotations

import hashlib

from mcp.server.fastmcp import FastMCP

from src.mcp.guardrails.patterns import scan_text
from src.schemas.agent_schemas import PIIScanResult, RedactedResult

mcp_server = FastMCP("guardrails")


def _get_replacement(pii_type: str, value: str, strategy: str) -> str:
    """Build the replacement string for a single PII detection."""
    if strategy == "mask":
        return f"[REDACTED_{pii_type}]"
    elif strategy == "hash":
        h = hashlib.sha256(value.encode()).hexdigest()[:8]
        return f"[{pii_type}:{h}]"
    elif strategy == "remove":
        return ""
    # Default to mask
    return f"[REDACTED_{pii_type}]"


@mcp_server.tool()
async def scan_pii(text: str) -> dict:
    """Scan text for PII patterns (PT/EU + international).

    Returns a dict with has_pii, detections list, and scanned_length.
    """
    detections = scan_text(text)
    result = PIIScanResult(
        has_pii=len(detections) > 0,
        detections=detections,
        scanned_length=len(text),
    )
    return result.model_dump()


@mcp_server.tool()
async def redact_text(text: str, strategy: str = "mask") -> dict:
    """Redact all detected PII from text.

    Strategies: "mask" (default), "hash", "remove".
    Returns a dict with original_length, redacted_text, redactions_applied, strategy.
    """
    detections = scan_text(text)

    # Apply redactions in reverse order to preserve character offsets
    redacted = text
    for det in sorted(detections, key=lambda d: d.start, reverse=True):
        replacement = _get_replacement(det.pii_type, det.value, strategy)
        redacted = redacted[: det.start] + replacement + redacted[det.end :]

    result = RedactedResult(
        original_length=len(text),
        redacted_text=redacted,
        redactions_applied=len(detections),
        strategy=strategy,
    )
    return result.model_dump()
