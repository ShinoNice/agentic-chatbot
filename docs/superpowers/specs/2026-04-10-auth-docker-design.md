# Auth + Docker Polish — Design Spec

**Date:** 2026-04-10
**Branch:** `claude/2026-04-10-auth-docker`
**Purpose:** Unblock Azure deployment by adding API key auth and hardening the Docker build.

---

## 1. API Key Auth

**Env var:** `API_KEY` (required — app refuses to start if missing).

**Mechanism:** FastAPI dependency `verify_api_key` in `src/api/auth.py`. Reads `X-API-Key` header, compares against `settings.api_key` using `secrets.compare_digest` (timing-safe). Returns 401 if missing or wrong.

**Route coverage:**
- `/api/chat` — protected
- `/api/ingest` — protected
- `/api/health` — open (health probes need it)

**Streamlit:** reads `API_KEY` from env, passes as `X-API-Key` header in every httpx call.

**Files:** `src/api/auth.py` (new), `src/core/config_loader.py`, `src/api/routes.py`, `ui/streamlit_frontend.py`, `.env.example`.

## 2. CORS Lockdown

Change `config/settings.yaml` default from `["*"]` to `["http://localhost:8501", "http://localhost:8001"]`. Override via env or YAML mount on Azure.

## 3. Docker: Multi-Stage + Non-Root + uv

**Stage 1 (builder):** `python:3.12-slim`, install uv, `uv sync --frozen --no-dev`.
**Stage 2 (runtime):** `python:3.12-slim`, copy venv + source, create `appuser` (non-root).

Un-gitignore `uv.lock` so `--frozen` works in the build context.

Bump `docker-compose.yml` health check `start_period` to 120s (reranker model download on first boot can take 30-60s).

## 4. Reranker Pre-Warming

In `src/api/app.py` `lifespan()`, after `try_connect_existing()`, call `reranker._ensure_loaded()` via `asyncio.to_thread` if rerank is enabled and the orchestrator exists.

## 5. Out of Scope

No rate limiting, no OAuth2/Entra ID, no CI/CD, no new tests.

## 6. Success Criteria

1. `/api/health` → 200 without key
2. `/api/chat` → 401 without key, works with key
3. `docker compose up --build` succeeds, `whoami` shows `appuser`
4. Streamlit UI works (auto-sends key)
