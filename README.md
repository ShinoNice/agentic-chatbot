# Agentic Chatbot

A multi-agent RAG (Retrieval-Augmented Generation) chatbot built with LangGraph, FastAPI and Streamlit.

## Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│  Streamlit UI   │ ────────────► │   FastAPI Backend │
│  (port 8501)    │   port 8001   │   (port 8001)     │
└─────────────────┘               └──────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              Vector Store           LLM (OpenAI)        MCP Servers
           (Pinecone / Chroma)        GPT-4o-mini     (Guardrails + Audit)
```

**LangGraph workflow:** Guardrails Input → Retrieve → Rerank → Relevance Check → Research → Guardrails Output → Verify

**Retrieval pipeline:** Hybrid search (BM25 + dense vectors) → Cross-encoder reranker (`BAAI/bge-reranker-base`) → top-K chunks → agents.

**MCP servers:** Two FastMCP servers integrated into the workflow for sensitive data handling:
- **Guardrails** — PII detection and redaction (PT/EU + international patterns) on both input queries and output answers
- **Audit Trail** — SQLite-backed event logging of every node transition for compliance

### Reranking

A cross-encoder reranking step sits between hybrid retrieval and the relevance check. Hybrid search returns a wide candidate pool (`rerank.candidate_k`, default 30); the reranker scores `(query, chunk)` pairs and keeps the top `rerank.top_k` (default 10). Toggleable via `config/settings.yaml`.

The chosen defaults are backed by a measured RAGAS sweep against `golden_set_v2.json`:

| Config | Context Precision | Context Recall | Faithfulness |
|---|---|---|---|
| Baseline (no rerank) | 0.673 | 0.967 | 0.951 |
| **30 → 10 (default)** | **0.923 (+25.0 pp)** | 1.000 (+3.3 pp) | 0.980 (+2.9 pp) |
| 30 → 5 (rejected) | 0.945 (+27.2 pp) | 0.900 (−6.7 pp) | 0.968 (+1.7 pp) |

The aggressive `top_k=5` config was rejected because it broke recall on comparison-style questions where the answer requires multiple chunks. Full sweep methodology, per-question delta analysis, and a discussion of the failure mode are in the [reranker design spec](docs/superpowers/specs/2026-04-08-reranker-design.md). Raw eval CSVs live under [evaluation/results/](evaluation/results/).

### MCP Servers

Two [Model Context Protocol](https://modelcontextprotocol.io/) servers provide sensitive-data capabilities, integrated as LangGraph nodes:

**Guardrails MCP** (`src/mcp/guardrails/`) — scans and redacts PII from both user queries (before retrieval) and draft answers (before delivery). Regex-based, zero API cost, fully deterministic.

Detected PII patterns:
- **Portuguese/EU:** NIF (mod-11 validated), PT phone (+351/9xx/2xx), NISS, Cartão de Cidadão, IBAN, Código Postal
- **International:** Email, credit card (Luhn validated), IPv4, date of birth, SSN, international phone

Three redaction strategies configurable via `config/settings.yaml`:
- `mask` (default) — `joao@test.pt` → `[REDACTED_EMAIL]`
- `hash` — `joao@test.pt` → `[EMAIL:a3f2b8...]`
- `remove` — deletes the matched text

**Audit Trail MCP** (`src/mcp/audit/`) — logs every LangGraph node transition (query received, documents retrieved, PII detected, answer generated, verification completed, etc.) to a SQLite database at `data/audit/audit.db`. Queryable via `GET /api/audit/{session_id}`. Configurable retention (default 90 days).

Both servers are built with FastMCP and can also run standalone for external MCP clients:
```bash
uv run python -m src.mcp.guardrails   # stdio MCP server
uv run python -m src.mcp.audit        # stdio MCP server
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — the project is uv-managed via `pyproject.toml` + `uv.lock`; pip/poetry/conda are not supported
- Docker & Docker Compose (for containerised run)
- API keys — copy `.env.example` to `.env` and fill in your keys

## Quick Start — Local

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/agentic-chatbot.git
cd agentic-chatbot

# 2. Install uv (if you don't have it)
# Windows (PowerShell):  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:         curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies (uv creates and populates .venv automatically)
uv sync

# 4. Configure secrets
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
# Edit .env and fill in your API keys

# 5. Run the API backend
uv run uvicorn src.api.app:app --reload --port 8001

# 6. Run the Streamlit UI (in a separate terminal)
uv run streamlit run ui/streamlit_frontend.py --server.port 8501

# 7. (Optional) Run the CLI
uv run python -m src.main
```

## Quick Start — Docker

```bash
# Copy and fill in secrets
copy .env.example .env   # then edit .env

# Build and start all services (first build takes ~5–10 min; layers cache afterwards)
docker compose up --build -d

# Confirm both containers are up and api is healthy
docker compose ps

# Services:
#   API      → http://localhost:8001
#   UI       → http://localhost:8501
#   API docs → http://localhost:8001/docs

# Tail logs / stop
docker compose logs -f
docker compose down
```

The image is a **multi-stage uv build** with CPU-only PyTorch wired at the package-manager level (`[tool.uv.sources]` in `pyproject.toml`), so `uv sync --frozen` in the builder stage produces a reproducible ~2–3 GB runtime image with no CUDA libraries, no compiler, and a non-root user. Inside the compose network, the UI reaches the API via the service name (`http://api:8001`) using the `API_BASE_URL` env var — host dev keeps using `http://localhost:8001`.

## Project Structure

```
├── config/          # YAML config (settings, prompts, logging)
├── data/            # Raw docs, cache, vector DB (gitignored)
├── evaluation/      # RAGAS / DeepEval evaluation pipeline
├── notebooks/       # Exploratory notebooks
├── src/
│   ├── api/         # FastAPI app, routes, schemas
│   ├── core/        # Config loader, logger, exceptions
│   ├── engines/     # OpenAI client, embedding model
│   ├── mcp/         # MCP servers (guardrails PII + audit trail)
│   ├── retrieval/   # Document processor, hybrid search, vector store
│   ├── schemas/     # Pydantic schemas
│   └── workflow/    # LangGraph orchestrator, agents, memory
├── tests/           # pytest suite (mirrors src/)
├── ui/              # Streamlit frontend
├── .env.example     # Template for required environment variables
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml   # Dependencies, tool config (uv-managed)
└── uv.lock          # Locked dependency graph (committed for reproducible builds)
```

## Environment Variables

See [.env.example](.env.example) for all required variables.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `PINECONE_API_KEY` | Pinecone key — leave empty to use local ChromaDB |
| `TAVILY_API_KEY` | Reserved for future web-search tool. **Currently unused** — no code in `src/` consumes it. |
| `LANGSMITH_API_KEY` | LangSmith observability (optional) |
| `GEMINI_API_KEY` | Google Gemini (optional, for secondary LLM) |
| `API_BASE_URL` | (Docker only) Set automatically by `docker-compose.yml` so the UI container reaches the API at `http://api:8001`. Unset for local dev. |
