"""Standalone entry point for the Audit Trail MCP server.

Run with: python -m src.mcp.audit
"""

from src.mcp.audit.server import mcp_server

mcp_server.run()
