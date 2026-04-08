# CLAUDE.md

This file provides guidance to Claude Code when working in this repository. The Projects2026 workflow kernel (`../CLAUDE.md`) is loaded above this file and defines branching, hooks, and intake rules — those rules apply here too.

## Project

**Agentic Chatbot** — a multi-agent RAG (Retrieval-Augmented Generation) chatbot that ingests PDF documents, retrieves relevant context via hybrid search, and generates verified answers through a self-correcting LangGraph workflow (Relevance Checker → Researcher → Verifier). Surfaces: FastAPI HTTP API, Streamlit web UI, and an interactive CLI.

**Purpose:** Portfolio / showcase project. Trade-off priority: **clarity, narrative, and visible quality** over enterprise-grade ops.

## Current Phase

**Tests + hardening.** The codebase functions end-to-end but has zero tests. Goal for the next several sessions: build a real test suite covering the agents, orchestrator, retrieval, and schemas — and fix tangled code as testing reveals it.

Permission granted to refactor existing files when writing tests exposes coupling that makes them hard to test cleanly. Refactors stay scoped to what's needed for the test in the same commit; no speculative cleanup.

## Environment

- **OS:** Windows 11
- **Shell:** Git Bash (use Unix shell syntax — `/dev/null`, forward slashes, `export`)
- **Python:** 3.12+
- **Package manager:** [uv](https://docs.astral.sh/uv/) (managed via `pyproject.toml` + `uv.lock`). Don't introduce pip/poetry/conda workflows.

## Tech Stack

| Layer | Technology |
|---|---|
| Workflow orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o-mini (via `langchain-openai`) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Retrieval | Hybrid: BM25 (sparse, 40%) + dense vector (60%) via LangChain `EnsembleRetriever` |
| Vector DB | Pinecone (cloud) **or** ChromaDB (local fallback) |
| PDF processing | Docling primary, PyMuPDF fallback |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAS (golden sets in `evaluation/datasets/`) |
| Observability | LangSmith (optional) |
| Config | Pydantic Settings + YAML |
| Tests | pytest + `unittest.mock` + monkeypatch |

## Architecture (brief)

The exhaustive architecture overview lives in [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md). Key files to know:

- [src/workflow/orchestrator.py](src/workflow/orchestrator.py) — `RAGOrchestrator`, the LangGraph `StateGraph` that wires the agents and the self-correction loop.
- [src/workflow/agents/relevance_checker.py](src/workflow/agents/relevance_checker.py), [researcher.py](src/workflow/agents/researcher.py), [verifier.py](src/workflow/agents/verifier.py) — the three agents.
- [src/workflow/memory.py](src/workflow/memory.py) — `AgentState` TypedDict (shared state).
- [src/retrieval/](src/retrieval/) — `document_processor.py`, `hybrid_search.py`, `vector_store.py`.
- [src/api/app.py](src/api/app.py), [routes.py](src/api/routes.py), [dependencies.py](src/api/dependencies.py) — FastAPI surface.
- [src/core/config_loader.py](src/core/config_loader.py) — Pydantic Settings, reads `config/settings.yaml` + `.env`.
- [src/engines/openai_client.py](src/engines/openai_client.py), [embedding_model.py](src/engines/embedding_model.py) — LLM/embedding wrappers behind `BaseLLM` / `BaseEmbeddingModel` ABCs.
- [ui/streamlit_frontend.py](ui/streamlit_frontend.py) — Streamlit UI.
- [evaluation/eval_pipeline.py](evaluation/eval_pipeline.py) — RAGAS evaluation script.

## Hard Constraints

1. **No paid API calls in the test suite.** Tests must mock OpenAI, Pinecone, and Tavily clients. No real network calls in `pytest`. If we ever want a real-API integration test, it must be opt-in via an env flag (e.g. `INTEGRATION=1`) and skipped by default.
2. **Windows-friendly tooling only.** No Linux/Mac-only commands in scripts, fixtures, or hooks. Paths and process management must work under Git Bash on Windows.
3. **Python 3.12+ via uv only.** Don't introduce tools or deps requiring older Python or non-uv package management.

## Conventions

- **Test layout:** `tests/` at repo root, mirroring `src/` package structure (e.g. `tests/workflow/test_orchestrator.py`). Shared fixtures in `tests/conftest.py`.
- **Mocking:** prefer pytest fixtures + `monkeypatch` for env vars; `unittest.mock.MagicMock` / `patch` for OpenAI / Pinecone / Chroma clients. If a test would benefit from recorded responses, use `vcrpy` — but mocks first.
- **Refactor discipline:** when a test reveals tangled code, refactor it in the **same commit** as the test that motivated it. Don't accumulate "cleanup later" debt; don't go on speculative refactor sprees.
- **Entry points (CLI / Streamlit / Docker) may briefly break during a refactor**, but every commit must leave them green. If a refactor crosses commit boundaries, fix the entry points before committing.
- **Commits stay focused:** one logical change per commit. Use the commit message body to explain *why*, not just *what*.

## Hook Commands

<!-- Parsed by Projects2026 hooks. Do not remove. -->

```
test_command: uv run pytest -q --no-header
lint_command: uv run ruff check .
dev_command: uvicorn src.api.app:app --reload --port 8001
```

The `p26-test-before-commit.sh` hook explicitly treats pytest exit code 5 ("no tests collected") as allowed, so this command works even before any tests exist. Once tests are added, pytest will return 0 on success and the same command will keep working.

## Open Questions / Things to Verify

- **Tavily web search wiring:** `TAVILY_API_KEY` is in `.env.example` but it's not yet confirmed whether any agent or tool actually uses it. Verify before writing tests for the workflow.
- **Project audit:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) section 14 ("Honest Project Audit") likely lists known issues in the existing code. Read it before planning the test strategy so we test the things that actually need testing.
- **Integration test gate:** decide whether to add an opt-in integration test that runs the orchestrator end-to-end against local Chroma + a tiny mock LLM, vs keeping everything as pure unit tests.

## Mandatory Skills (inherited from Projects2026 kernel)

These are not optional. Invoke them **before** the situation they cover:

- `superpowers:brainstorming` — before any feature/design work
- `superpowers:writing-plans` — before any multi-step implementation
- `superpowers:test-driven-development` — before writing implementation code
- `superpowers:systematic-debugging` — for any bug or unexpected behavior
- `superpowers:verification-before-completion` — before claiming anything done
- `superpowers:requesting-code-review` — before merging a branch
