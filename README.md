# Agentic Chatbot

> Self-correcting multi-agent RAG on user-uploaded PDFs, with MCP-based PII guardrails and a full audit trail. Live on Azure Container Apps.

**🌐 Live demo:** <https://ca-chatbot-ui.blacktree-3305419b.northeurope.azurecontainerapps.io/>
Upload any PDF, ask it questions — the chatbot answers only from your document, with a self-verifying answer loop.

![CI](https://github.com/ShinoNice/agentic-chatbot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

<!-- Drop a screenshot or GIF of the Streamlit UI here once captured:
![UI screenshot](docs/screenshots/ui-demo.png) -->

---

## What makes this different

- **Self-correcting agent loop (LangGraph).** A Relevance Checker → Researcher → Verifier pipeline re-runs retrieval when the verifier flags unsupported claims. Bounded by `app.max_iterations`, so it cannot loop forever.
- **Measured retrieval quality.** Hybrid BM25 + dense retrieval with a cross-encoder reranker lifts Context Precision from **0.673 → 0.923 (+25.0 pp)** on a 200-item RAGAS golden set — not a hand-wavy "LangChain demo" claim, a reproducible number.
- **Per-session PDF upload.** Visitors drop their own PDF; it's ingested into a session-scoped Pinecone namespace and isolated from the default corpus and from every other visitor. No cross-contamination.
- **Production-flavoured compliance primitives via MCP.** Two FastMCP servers wired into the graph: regex-based PII guardrails (12 PT/EU + international patterns with checksum validation) and a SQLite audit trail keyed by session ID, queryable at `GET /api/audit/{session_id}`.
- **Shipped.** Live on Azure Container Apps (UI public, API internal-only, secrets from Key Vault), documented in a real 288-line deployment runbook.

## Benchmarks (golden\_set\_v2, 200 questions, GPT-4o-mini)

| Config                    | Context Precision     | Context Recall   | Faithfulness     |
| ------------------------- | --------------------- | ---------------- | ---------------- |
| Baseline (no rerank)      | 0.673                 | 0.967            | 0.951            |
| **30 → 10 (default)**     | **0.923 (+25.0 pp)**  | 1.000 (+3.3 pp)  | 0.980 (+2.9 pp)  |
| 30 → 5 (rejected)         | 0.945 (+27.2 pp)      | 0.900 (−6.7 pp)  | 0.968 (+1.7 pp)  |

The aggressive `top_k=5` config was rejected because it broke recall on comparison-style questions where the answer requires multiple chunks. Full sweep methodology, per-question delta analysis, and a discussion of the failure mode are in the [reranker design spec](docs/superpowers/specs/2026-04-08-reranker-design.md). Raw CSVs: [evaluation/results/](evaluation/results/).

## Architecture

```
┌─────────────────┐     HTTPS     ┌──────────────────┐
│  Streamlit UI   │ ────────────► │   FastAPI Backend │
│   (port 8501)   │               │   (port 8001)     │
└─────────────────┘               └──────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              Vector Store           LLM (OpenAI)         MCP Servers
           (Pinecone / Chroma)       GPT-4o-mini      (Guardrails + Audit)
```

**LangGraph flow:** `guardrails_input → retrieve → rerank → check_relevance → research → guardrails_output → verify` (with a retry edge back to `retrieve` when the verifier rejects the draft answer).

**Retrieval:** Hybrid search (BM25 40% + dense 60% via `EnsembleRetriever`) fans out to `rerank.candidate_k=30` candidates, then `BAAI/bge-reranker-base` picks the top `rerank.top_k=10`.

## MCP Servers

Two [Model Context Protocol](https://modelcontextprotocol.io/) servers wired in as LangGraph nodes. They can also run standalone over stdio for external clients (Claude Desktop, etc.):

### Guardrails MCP — [src/mcp/guardrails/](src/mcp/guardrails/)

Regex-based PII detection + redaction on both input queries (pre-retrieval) and draft answers (pre-delivery). Zero API cost, deterministic, 36 pattern/redaction tests covering true AND false positives:

- **Portuguese/EU:** NIF (mod-11 validated), PT phone (+351/9xx/2xx), NISS, Cartão de Cidadão, IBAN, Código Postal
- **International:** Email, credit card (Luhn validated), IPv4, date of birth, SSN, international phone

Three redaction strategies: `mask` (default), `hash`, `remove`.

### Audit Trail MCP — [src/mcp/audit/](src/mcp/audit/)

SQLite-backed event store (via `aiosqlite`) logging every LangGraph node transition — query received, documents retrieved, PII detected, answer generated, verification completed. Retention configurable (default 90 days). Queryable via `GET /api/audit/{session_id}`.

```bash
# Run either server standalone as an MCP stdio server:
uv run python -m src.mcp.guardrails
uv run python -m src.mcp.audit
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — pip/poetry/conda are not supported
- Docker & Docker Compose (optional — for containerised run)

## Quick Start — Local

```bash
git clone https://github.com/ShinoNice/agentic-chatbot.git
cd agentic-chatbot

# Install uv (one-off, skip if already installed)
# Windows (PowerShell):  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:         curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
cp .env.example .env        # then fill in OPENAI_API_KEY (required)

# Run API + UI in two terminals
uv run uvicorn src.api.app:app --reload --port 8001
uv run streamlit run ui/streamlit_frontend.py --server.port 8501

# Optional: CLI
uv run python -m src.main
```

## Quick Start — Docker

```bash
cp .env.example .env
docker compose up --build -d

# Services:
#   API      → http://localhost:8001
#   UI       → http://localhost:8501
#   API docs → http://localhost:8001/docs
```

Multi-stage image, CPU-only PyTorch pinned via `[tool.uv.sources]`, non-root user, no `build-essential` at runtime.

## Testing

```bash
uv run pytest -q --no-header              # 94 tests
make cov                                  # with 60% coverage floor
uv run ruff check .                        # lint
```

CI (GitHub Actions) enforces both on every push + PR. Pre-commit hooks match.

## API surface

| Endpoint                      | Purpose                                                       |
| ----------------------------- | ------------------------------------------------------------- |
| `POST /api/chat`              | Ask a question (routes to session orchestrator if one exists) |
| `POST /api/upload`            | Upload a PDF (≤ 20 MiB) into a session-scoped namespace       |
| `POST /api/ingest`            | Ingest PDFs from `data/raw/` into the default corpus          |
| `GET  /api/audit/{sid}`       | Replay the audit trail for a session                          |
| `GET  /api/healthz`           | Liveness probe (always 200 while the loop is responsive)      |
| `GET  /api/readyz`            | Readiness probe (503 until the knowledge base is loaded)      |

Every response carries an `X-Request-ID` header; logs surface the same ID so a single chat turn is reconstructable end-to-end.

## Project Structure

```
├── .github/workflows/ # CI (tests + ruff + Docker smoke build)
├── config/            # YAML config (settings, prompts, logging)
├── data/              # Raw docs, cache, vector DB (gitignored)
├── docs/              # Azure deployment runbook, design specs
├── evaluation/        # RAGAS evaluation pipeline + golden sets
├── infra/             # Bicep IaC for the Azure stack
├── src/
│   ├── api/           # FastAPI app, routes, schemas, middleware
│   ├── core/          # Config loader, logger, exceptions
│   ├── engines/       # OpenAI client, embedding model
│   ├── mcp/           # MCP servers (guardrails PII + audit trail)
│   ├── retrieval/     # Document processor, hybrid search, vector store, reranker
│   ├── schemas/       # Pydantic schemas
│   └── workflow/      # LangGraph orchestrator, agents, memory
├── tests/             # pytest suite (mirrors src/, 67% coverage)
├── ui/                # Streamlit frontend
└── pyproject.toml     # Dependencies, tool config (uv-managed)
```

## Environment Variables

See [.env.example](.env.example) for the full template.

| Variable              | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| `OPENAI_API_KEY`      | OpenAI API key (required)                                                 |
| `PINECONE_API_KEY`    | Pinecone key — leave empty to fall back to local ChromaDB                 |
| `LANGSMITH_API_KEY`   | LangSmith observability (optional)                                        |
| `LANGSMITH_TRACING`   | Set to `true` to enable LangSmith tracing on both CLI and API paths       |
| `API_BASE_URL`        | Docker-compose sets this so the UI container reaches the API by service DNS |

## Deployment

The live demo runs on Azure Container Apps. One-shot deployment via Bicep:

```bash
az deployment group create \
  --resource-group rg-agentic-chatbot \
  --template-file infra/main.bicep \
  --parameters @infra/main.parameters.json
```

Full deployment notes (including the ACA internal/public split, Key Vault wiring, and the quirks of `az containerapp update` CLI arg parsing) live in [docs/azure-deployment-notes.md](docs/azure-deployment-notes.md).

## License

[MIT](LICENSE)
