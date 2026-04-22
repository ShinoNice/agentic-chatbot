# syntax=docker/dockerfile:1.7

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

# Docling transitively depends on OpenCV / Qt / Pillow bindings that try to
# open X11 libraries (libxcb.so.1) even on headless servers. Without these the
# parser crashes at load time with:
#     libxcb.so.1: cannot open shared object file: No such file or directory
# and the pipeline silently falls back to PyMuPDF, losing Docling's structured
# output. Install just the minimum X-less shims; no X server is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libxext6 \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home --home-dir /home/appuser appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY ui/ ./ui/
COPY config/ ./config/
COPY evaluation/ ./evaluation/
COPY pyproject.toml ./

# BM25 cache: .dockerignore excludes data/ by default to keep the build
# context small. If you want a populated BM25 index baked into the image
# (eliminates the dense-only degradation in ACA), uncomment the line below
# AND remove `data/cache/*.json` from .dockerignore for the subset you want
# shipped. Keep individual chunk caches out — they can be gigabytes.
# COPY data/cache/bm25_documents.json ./data/cache/bm25_documents.json

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face / sentence-transformers cache location. Pinning it inside the
# image lets us pre-bake the reranker model in the next RUN step.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Pre-download the cross-encoder reranker so the first user query doesn't
# eat a ~3-10s cold-download per replica start. Model is ~278 MB; the
# image grows by that amount but first-query latency drops to <1s.
RUN mkdir -p "$HF_HOME" \
    && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

RUN mkdir -p /app/data /app/data/logs /app/data/cache /app/data/audit \
    && chown -R appuser:appgroup /app/data /app/.cache

USER appuser

EXPOSE 8001
EXPOSE 8501
