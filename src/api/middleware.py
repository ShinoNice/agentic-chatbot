"""ASGI middleware: per-request correlation IDs.

A single ``X-Request-ID`` is generated (or propagated from upstream) for every
HTTP request and made available to any log record in the same request via a
``contextvars.ContextVar``. Structured logs can then pull this ID off the
current context and emit it under a stable field name, so a single chat
turn's 5-node LangGraph journey can be reconstructed from logs alone.

Middleware is deliberately minimal — no third-party deps beyond starlette (a
FastAPI transitive), no logging of bodies, no side effects beyond the header.
"""

from __future__ import annotations

import contextvars
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

_REQUEST_ID_HEADER = "X-Request-ID"

# Thread/task-local storage so nested code (logger formatters, LangGraph nodes,
# downstream LLM calls) can retrieve the current request's ID without passing
# it as an explicit argument. ``None`` means "no active request."
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)


def get_request_id() -> str | None:
    """Return the current request's correlation ID, or None outside a request."""
    return request_id_var.get()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a stable X-Request-ID to every request/response pair.

    - If the client supplies ``X-Request-ID``, we reuse it (lets an ingress
      such as Azure Front Door or an upstream proxy propagate its own IDs).
    - Otherwise we mint a new UUID4.
    - The ID is stored in a ``ContextVar`` for the lifetime of the request so
      log records produced from deep in the stack can surface it.
    """

    def __init__(self, app: ASGIApp, header_name: str = _REQUEST_ID_HEADER) -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(self.header_name)
        request_id = incoming or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.header_name] = request_id
        return response
