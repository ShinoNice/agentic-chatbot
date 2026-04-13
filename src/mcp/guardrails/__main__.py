"""Standalone entry point for the Guardrails MCP server.

Run with: python -m src.mcp.guardrails
"""

from src.mcp.guardrails.server import mcp_server

mcp_server.run()
