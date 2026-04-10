"""API key authentication dependency for FastAPI routes."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.core.config_loader import settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
) -> str:
    """Validate the X-API-Key header against the configured API_KEY.

    Uses ``secrets.compare_digest`` to prevent timing attacks.
    Returns the validated key so downstream code can log it if needed.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return api_key
