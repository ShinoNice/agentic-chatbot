# CLAUDE.md

This file provides guidance to Claude Code when working in this repository. The Projects2026 workflow kernel (`../CLAUDE.md`) is loaded above this file and defines branching, hooks, and intake rules — those rules apply here too.

## Project

**Agentic Chatbot** — a multi-agent RAG (Retrieval-Augmented Generation) chatbot that ingests PDF documents, retrieves relevant context via hybrid search, and generates verified answers through a self-correcting LangGraph workflow (Relevance Checker → Researcher → Verifier). Surfaces: FastAPI HTTP API, Streamlit web UI, and an interactive CLI.

**Purpose:** Portfolio / showcase project. Trade-off priority: **clarity, narrative, and visible quality** over enterprise-grade ops.

## Current Phase

**Retrieval quality (one-branch detour from "Tests + hardening").** A cross-encoder reranker (`BAAI/bge-reranker-base`) was added between hybrid retrieval and the relevance check. Measured RAGAS lift on `golden_set_v2.json`: Context Precision **+24.96 pp** (0.673 → 0.923), Context Recall **+3.33 pp**, Faithfulness **+2.92 pp**, Answer Relevancy **+1.51 pp**. Default config: `candidate_k=30, top_k=10`. The aggressive `top_k=5` config was rejected because it broke recall on comparison-style questions. See [docs/superpowers/specs/2026-04-08-reranker-design.md §13](docs/superpowers/specs/2026-04-08-reranker-design.md) for the full results and per-question analysis. Raw eval CSVs in [evaluation/results/](evaluation/results/).

**Test suite has bootstrapped:** 10 reranker unit tests pass under `uv run pytest -q --no-header`. The `p26-test-before-commit.sh` hook's "no tests collected" grace period is over — every commit on this branch and onward must keep pytest exit 0.

**Next session: tests + hardening resumes.** The reranker module is the only thing with tests. The agents, orchestrator, retrieval pipeline, and schemas are still untested.

Permission granted to refactor existing files when writing tests exposes coupling that makes them hard to test cleanly. Refactors stay scoped to what's needed for the test and are staged in the same chunk you hand to the user for commit; no speculative cleanup.

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
- **Refactor discipline:** when a test reveals tangled code, stage the refactor together with the test that motivated it as a single coherent chunk for the user to commit. Don't accumulate "cleanup later" debt; don't go on speculative refactor sprees.
- **Entry points (CLI / Streamlit / Docker) may briefly break during a refactor**, but every chunk handed to the user for commit must leave them green. If a refactor crosses chunk boundaries, fix the entry points before staging the chunk.
- **Staged chunks stay focused:** one logical change per chunk you hand to the user. When you suggest a commit message for the user, use the message body to explain *why*, not just *what* — and never include `Co-Authored-By` trailers (kernel Section 0).

## Hook Commands

<!-- Parsed by Projects2026 hooks. Do not remove. -->

```
test_command: uv run pytest -q --no-header
lint_command: uv run ruff check .
dev_command: uvicorn src.api.app:app --reload --port 8001
```

The `p26-test-before-commit.sh` hook previously treated pytest exit code 5 ("no tests collected") as allowed. **As of the reranker branch, the test suite has bootstrapped (10 reranker tests) and the grace period is over** — every commit must keep `uv run pytest -q --no-header` returning exit 0.

## Open Questions / Things to Verify

- **Tavily web search wiring:** `TAVILY_API_KEY` is in `.env.example` but it's not yet confirmed whether any agent or tool actually uses it. Verify before writing tests for the workflow.
- **Project audit:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) section 14 ("Honest Project Audit") lists known issues. Read it before planning the test strategy. The reranker branch partially fixed §14.2's "sync I/O in async paths" by wrapping `CrossEncoder.predict()` in `asyncio.to_thread`; existing offenders (BM25 cache I/O, blocking `input()` in CLI) are still untouched.
- **Integration test gate:** still open. The reranker branch chose unit tests only (mocking the cross-encoder via monkeypatch) for the new module. The broader question — opt-in `INTEGRATION=1` end-to-end test against local Chroma + stubbed LLM — is for the next phase.
- **Reranker pre-warming:** the BGE model is lazy-loaded on the first query (~3-10s cold start). For local dev this is fine; in deployed contexts a `lifespan` warm-up call would smooth it. Not blocking; deferred.
- **Wider candidate-pool sweep:** the eval sweep was reduced from 5 configs to 3 (baseline / 30→10 / 30→5). The 50→{10,5} arms were not run. If a future eval shows the chosen `candidate_k=30` is leaving precision on the table, run those.

## Mandatory Skills (inherited from Projects2026 kernel)

These are not optional. Invoke them **before** the situation they cover:

- `superpowers:brainstorming` — before any feature/design work
- `superpowers:writing-plans` — before any multi-step implementation
- `superpowers:test-driven-development` — before writing implementation code
- `superpowers:systematic-debugging` — for any bug or unexpected behavior
- `superpowers:verification-before-completion` — before claiming anything done
- `superpowers:requesting-code-review` — before telling the user a branch is ready to merge
