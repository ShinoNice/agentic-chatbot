# ─── Stage 1: Builder ────────────────────────────────────────────────────────
# Install dependencies into an isolated venv using uv. Dev deps excluded.
# CPU-only torch is configured in pyproject.toml via [tool.uv.sources],
# so --frozen uses the lockfile directly with no CUDA wheels.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --no-install-project --frozen

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
# Slim image with only the venv + source. No compiler, no dev deps.
FROM python:3.12-slim

RUN groupadd --system appgroup && useradd --system --gid appgroup appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY ui/ ./ui/
COPY config/ ./config/
COPY evaluation/ ./evaluation/
COPY pyproject.toml ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/data && chown -R appuser:appgroup /app/data

USER appuser

EXPOSE 8001
EXPOSE 8501
