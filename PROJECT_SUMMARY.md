# Agentic Chatbot - Project Summary

A **multi-agent RAG (Retrieval-Augmented Generation) chatbot** that ingests PDF documents, retrieves relevant context via hybrid search, and generates verified answers through a self-correcting agent workflow. Built with **LangGraph**, **FastAPI**, and **Streamlit**.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Multi-Agent Workflow (LangGraph)](#3-multi-agent-workflow-langgraph)
4. [Retrieval Pipeline](#4-retrieval-pipeline)
5. [API Layer (FastAPI)](#5-api-layer-fastapi)
6. [Frontend (Streamlit)](#6-frontend-streamlit)
7. [LLM & Embedding Engines](#7-llm--embedding-engines)
8. [Configuration](#8-configuration)
9. [Evaluation Pipeline](#9-evaluation-pipeline)
10. [Deployment](#10-deployment)
11. [Development Tools](#11-development-tools)
12. [Dependencies](#12-dependencies)
13. [Data](#13-data)
14. [Honest Project Audit](#14-honest-project-audit)
15. [Scaling & Production Deployment on Microsoft Azure](#15-scaling--production-deployment-on-microsoft-azure)
16. [Audit Methodology](#audit-methodology)

---

## 1. Architecture Overview

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│  Streamlit UI   │ ────────────► │   FastAPI Backend │
│  (port 8501)    │   port 8001   │   (port 8001)     │
└─────────────────┘               └──────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              Vector Store           LLM (OpenAI)         Web Search
           (Pinecone / Chroma)        GPT-4o-mini          (Tavily)
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Workflow Orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Retrieval | Hybrid: BM25 (sparse) + Vector (dense) |
| Vector Database | Pinecone (cloud) or ChromaDB (local) |
| PDF Processing | Docling + PyMuPDF (fallback) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| API Framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Observability | LangSmith (optional) |
| Config | Pydantic + YAML |

### End-to-End Data Flow

```
User Question (UI / CLI / API)
        │
        ▼
   FastAPI Route (/api/chat)
        │
        ▼
   SystemManager.query(question)
        │
        ▼
   RAGOrchestrator.run(question)   ← LangGraph StateGraph
        │
   ┌────┴────────────────────────────────────────────┐
   │ [1] RETRIEVE                                    │
   │     HybridSearcher → EnsembleRetriever          │
   │     (40% BM25 + 60% Vector) → top-k docs       │
   │                                                 │
   │ [2] CHECK RELEVANCE                             │
   │     RelevanceCheckerAgent → structured output   │
   │     CAN_ANSWER / PARTIAL → proceed              │
   │     NO_MATCH → stop, return "cannot answer"     │
   │                                                 │
   │ [3] RESEARCH                                    │
   │     ResearchAgent → draft answer with citations │
   │                                                 │
   │ [4] VERIFY                                      │
   │     VerificationAgent → checks hallucinations   │
   │     supported → finalize                        │
   │     unsupported & iterations < 3 → retry [3]    │
   └─────────────────────────────────────────────────┘
        │
        ▼
   ChatResponse (answer, sources, verification, iterations)
```

---

## 2. Project Structure

```
agentic-chatbot/
├── config/                             # Configuration files
│   ├── settings.yaml                   # LLM, RAG, Docling, and app settings
│   ├── logging_config.yaml             # Logging config (console + JSON file)
│   └── prompts/
│       └── agent_system.yaml           # System prompts for agents
├── data/                               # Data directory (gitignored)
│   ├── raw/                            # Source PDFs for ingestion
│   ├── cache/                          # Cached document chunks (JSON)
│   ├── vector_db/                      # ChromaDB persistence directory
│   └── logs/                           # Application logs
├── src/                                # Main source code
│   ├── main.py                         # CLI entry point (interactive REPL)
│   ├── api/
│   │   ├── app.py                      # FastAPI app creation + lifespan
│   │   ├── routes.py                   # Endpoints: /health, /ingest, /chat
│   │   ├── schemas.py                  # Pydantic request/response models
│   │   └── dependencies.py             # SystemManager singleton
│   ├── core/
│   │   ├── config_loader.py            # Settings from YAML + .env (Pydantic)
│   │   ├── logger.py                   # Logging setup from YAML
│   │   └── exceptions.py               # Custom exception hierarchy
│   ├── engines/
│   │   ├── base.py                     # Abstract BaseLLM, BaseEmbeddingModel
│   │   ├── openai_client.py            # OpenAI ChatOpenAI wrapper
│   │   └── embedding_model.py          # OpenAI embeddings wrapper
│   ├── retrieval/
│   │   ├── document_processor.py       # PDF parsing, chunking, caching
│   │   ├── vector_store.py             # Pinecone / ChromaDB manager
│   │   └── hybrid_search.py            # Ensemble retriever (vector + BM25)
│   ├── schemas/
│   │   └── agent_schemas.py            # RelevanceStatus, VerificationReport, etc.
│   └── workflow/
│       ├── orchestrator.py             # RAGOrchestrator: LangGraph graph
│       ├── memory.py                   # AgentState TypedDict (shared state)
│       └── agents/
│           ├── relevance_checker.py    # Audits document relevance
│           ├── researcher.py           # Generates draft answers
│           └── verifier.py             # QA audit for hallucinations
├── ui/
│   └── streamlit_frontend.py           # Streamlit web UI
├── evaluation/
│   ├── eval_pipeline.py                # RAGAS evaluation script
│   └── datasets/
│       ├── golden_set_v1.json          # Test dataset v1 (~25 Q&A pairs)
│       └── golden_set_v2.json          # Test dataset v2 (~32 Q&A pairs)
├── misc/                               # Reference & utility files
│   ├── generate_workflow_diagram.py    # LangGraph Mermaid diagram generator
│   ├── env_setup.txt                   # UV/Python setup guide (Windows)
│   ├── git.txt                         # Git workflow guide
│   ├── terminal.txt                    # Terminal command reference
│   ├── results_eval.txt                # Evaluation results log
│   └── workflow_diagram*.png           # Generated workflow diagrams
├── main.py                             # Root entry: starts uvicorn server
├── .env.example                        # Template for environment variables
├── pyproject.toml                      # Project metadata & dependencies
├── requirements.txt                    # Pinned dependency list
├── Dockerfile                          # Multi-service Docker image
├── docker-compose.yml                  # Orchestrates API + UI containers
├── MAKEFILE                            # Development shortcuts (26 targets)
└── uv.lock                             # UV lock file
```

---

## 3. Multi-Agent Workflow (LangGraph)

The core intelligence is a **LangGraph StateGraph** with four nodes and conditional routing. The orchestrator lives in `src/workflow/orchestrator.py`.

### Graph Structure

```
START → retrieve → check_relevance ──┐
                                     ├── (NO_MATCH) → END
                                     └── (CAN_ANSWER/PARTIAL) → research → verify ──┐
                                                                  ▲                  │
                                                                  │    (unsupported  │
                                                                  │    & iter < 3)   │
                                                                  └──── retry ───────┘
                                                                         │
                                                                    (supported)
                                                                         │
                                                                         ▼
                                                                        END
```

### Agent State (`src/workflow/memory.py`)

All agents share a typed state dictionary:

| Field | Type | Purpose |
|-------|------|---------|
| `question` | `str` | The user's query |
| `documents` | `List[Document]` | Retrieved context (deduplicated via merge reducer) |
| `relevance_status` | `RelevanceStatus` | CAN_ANSWER, PARTIAL, or NO_MATCH |
| `draft_answer` | `Optional[str]` | Answer from the research agent |
| `verification` | `Optional[VerificationReport]` | QA audit result |
| `iterations` | `int` | Self-correction loop counter (max 3) |
| `error` | `Optional[str]` | Error message if something fails |

### Agent Descriptions

#### Relevance Checker (`src/workflow/agents/relevance_checker.py`)
- **Role:** Determines if retrieved documents can answer the question.
- **Input:** Question + retrieved documents (max 12,000 chars).
- **Output:** `RelevanceResponse` (structured output) containing a `RelevanceStatus` enum and reasoning.
- **Statuses:**
  - `CAN_ANSWER` — context directly answers the question.
  - `PARTIAL` — context mentions the topic but lacks full details.
  - `NO_MATCH` — context is unrelated to the question.
- **System Persona:** "Strict Relevance Auditor."

#### Research Agent (`src/workflow/agents/researcher.py`)
- **Role:** Synthesizes retrieved documents into a draft answer.
- **Input:** Question + documents (formatted as numbered "CONTEXT BLOCK" sections).
- **Output:** Draft answer string with citations in `[Source Name]` format.
- **Constraints:** Must use ONLY provided context. Minimum 5-character response.
- **System Persona:** "Precise Research Assistant."

#### Verification Agent (`src/workflow/agents/verifier.py`)
- **Role:** QA audit — checks the draft answer for hallucinations and contradictions.
- **Input:** Question + draft answer + documents.
- **Output:** `VerificationReport` (structured output):
  - `supported: bool` — whether all claims are backed by context.
  - `unsupported_claims: List[str]` — claims not found in context.
  - `contradictions: List[str]` — claims that contradict the context.
  - `relevant: bool` — whether the answer addresses the question.
  - `additional_details: Optional[str]` — extra notes.
- **System Persona:** "Strict QA Auditor."

### Conditional Routing Logic

1. **After Relevance Check:**
   - `NO_MATCH` → stop (return "cannot answer from available documents").
   - `CAN_ANSWER` or `PARTIAL` → proceed to research.

2. **After Verification:**
   - `supported == True` → finalize (return answer).
   - `supported == False` AND `iterations < max_iterations (3)` → retry research.
   - `supported == False` AND `iterations >= 3` → finalize with current answer anyway.

---

## 4. Retrieval Pipeline

### Document Processing (`src/retrieval/document_processor.py`)

Handles PDF ingestion with a dual-parser strategy:

1. **Primary: Docling** — structured extraction with optional OCR (RapidOCR).
   - Timeout: 300 seconds per document.
   - Skipped for PDFs > 80 pages (performance).
   - Falls back if fewer than 5 chunks produced.
2. **Fallback: PyMuPDF (fitz)** — fast text extraction.
3. **Chunking:** `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap).
4. **Deduplication:** SHA256 hash of `filename + content` per chunk.
5. **Metadata:** Each chunk carries `source`, `page_number`, `chunk_index`, `chunk_hash`, `parser`.
6. **Caching:** Processed chunks saved as JSON in `data/cache/` to avoid reprocessing.

### Vector Store (`src/retrieval/vector_store.py`)

Manages document storage with two backends:

| Backend | When Used | Storage | Details |
|---------|-----------|---------|---------|
| **Pinecone** | If `PINECONE_API_KEY` is set | Cloud (AWS us-east-1) | Serverless, cosine similarity, 1536-dim |
| **ChromaDB** | Default fallback | Local (`data/vector_db/`) | Persistent, no cloud dependency |

- Uses `chunk_hash` as document IDs for upsert deduplication.
- Metadata cleaning: flattens/serializes complex types for Pinecone compatibility.

### Hybrid Search (`src/retrieval/hybrid_search.py`)

Combines two retrieval strategies via `EnsembleRetriever`:

| Strategy | Weight | Type | Strengths |
|----------|--------|------|-----------|
| **BM25** | 0.4 (40%) | Sparse / keyword | Exact term matching, acronyms |
| **Vector** | 0.6 (60%) | Dense / semantic | Meaning-based, paraphrases |

- Returns `top_k` documents (default 10).
- BM25 index cached to `data/cache/bm25_documents.json`.
- Rebuilt on each ingestion.

---

## 5. API Layer (FastAPI)

### Application Setup (`src/api/app.py`)

- Created via `create_app()` factory function.
- **Lifespan hook:** Initializes `SystemManager` on startup; attempts to connect to existing vector store.
- **CORS:** Permissive (allows all origins) for local development.

### SystemManager (`src/api/dependencies.py`)

Central singleton orchestrating the entire RAG lifecycle:

| Property/Method | Description |
|-----------------|-------------|
| `is_ready` | Boolean — whether the orchestrator is initialized |
| `vector_store_type` | `"pinecone"` or `"chromadb"` |
| `documents_indexed` | Count of chunks in the vector store |
| `ingest(namespace)` | Processes all PDFs from `data/raw/`, builds vector store & BM25 |
| `query(question)` | Executes the full agentic workflow, returns results dict |
| `try_connect_existing()` | Connects to an already-populated vector store (skip re-ingestion) |

### Endpoints (`src/api/routes.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health: status, knowledge_base_ready, vector_store_type, documents_indexed |
| `/api/ingest` | POST | Ingest PDFs from `data/raw/`. Body: `{ namespace: "default" }`. Returns files processed + total chunks |
| `/api/chat` | POST | Query the knowledge base. Body: `{ question, session_id? }`. Returns answer, relevance, verification, sources, iterations |

### Request/Response Schemas (`src/api/schemas.py`)

**ChatRequest:**
- `question`: string (1-2000 characters)
- `session_id`: optional string

**ChatResponse:**
- `answer`: string
- `session_id`: optional string
- `relevance_status`: `"CAN_ANSWER"` | `"PARTIAL"` | `"NO_MATCH"`
- `verification`: `{ supported, unsupported_claims, contradictions, relevant, additional_details }`
- `sources`: list of `{ source, page_number, snippet }` (first 200 chars of each chunk)
- `iterations`: int (number of self-correction loops)

**IngestResponse:**
- `files_processed`: list of filenames
- `total_chunks`: int
- `status`: `"pending"` | `"completed"` | `"failed"`
- `message`: string

**HealthResponse:**
- `status`: string
- `knowledge_base_ready`: bool
- `vector_store_type`: string
- `documents_indexed`: int

---

## 6. Frontend (Streamlit)

**File:** `ui/streamlit_frontend.py`

### Features

- **Chat Interface:** Conversational UI with message history.
- **Sidebar Controls:**
  - API URL configuration (default: `http://localhost:8001`).
  - Health check button (shows system readiness).
  - Document ingestion trigger.
  - Clear chat button.
- **Rich Responses:**
  - Verification badges (supported / unsupported warnings).
  - Lists of unsupported claims and contradictions when detected.
  - Expandable source documents (deduplicated, with page numbers and snippets).
  - Iteration counter showing self-correction attempts.
- **HTTP Client:** Uses `httpx` with 120-second timeout.

---

## 7. LLM & Embedding Engines

### Base Classes (`src/engines/base.py`)

- `BaseLLM` — abstract interface for LLM implementations.
- `BaseEmbeddingModel` — abstract interface for embedding models.

### OpenAI Client (`src/engines/openai_client.py`)

- **Class:** `OpenAIClient(BaseLLM)`
- **Wraps:** `langchain_openai.ChatOpenAI`
- **Model:** `gpt-4o-mini` (configurable)
- **Temperature:** 0.2 (deterministic)
- **Max Tokens:** 2000
- **Methods:**
  - `async generate(system_prompt, user_prompt, model_name)` — async LLM call.
  - `get_model()` — returns raw `ChatOpenAI` instance for `.with_structured_output()`.

### Embedding Model (`src/engines/embedding_model.py`)

- **Class:** `OpenAIEmbeddingModel(BaseEmbeddingModel)`
- **Model:** `text-embedding-3-small` (1536 dimensions)
- **Wraps:** `langchain_openai.OpenAIEmbeddings`
- **Factory:** `get_embedding_engine()` returns the model instance.

---

## 8. Configuration

### Settings (`config/settings.yaml` + `src/core/config_loader.py`)

All configuration is loaded from YAML and `.env`, validated via Pydantic:

```yaml
model_settings:
  primary_llm: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  temperature: 0.2
  max_tokens: 2000

rag_settings:
  raw_data_dir: "data/raw"
  cache_dir: "data/cache"
  vector_db_path: "data/vector_db"
  pinecone_index_name: "ai-multiagent-index"
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 10
  hybrid_weights: [0.4, 0.6]   # [BM25, Vector]

docling_settings:
  do_ocr: false
  force_full_page_ocr: false
  images_scale: 1.0
  min_chunks_fallback: 5

app:
  debug_mode: true
  max_iterations: 3
  cors_origins: ["*"]
```

### Pydantic Settings Classes

```
Settings (root)
├── LLMSettings     → primary_llm, embedding_model, temperature, max_tokens
├── RAGSettings     → chunk_size, chunk_overlap, top_k, hybrid_weights, paths
├── DoclingSettings → do_ocr, force_full_page_ocr, images_scale, min_chunks_fallback
├── AppSettings     → debug_mode, max_iterations, cors_origins
└── prompts: dict   → loaded from config/prompts/*.yaml
```

### Agent System Prompts (`config/prompts/agent_system.yaml`)

Three agent personas defined:
- **researcher:** "Precise Research Assistant" — answers from context only.
- **verifier:** "Strict QA Auditor" — verifies answers against context.
- **relevance_checker:** "Strict Relevance Auditor" — determines if context answers question.

### Logging (`config/logging_config.yaml`)

- **Console Handler:** INFO level, standard format.
- **File Handler:** DEBUG level, JSON format, rotating (10 MB max, 5 backups).
- **Output:** `data/logs/app.json.log`
- **Suppresses:** Noisy loggers from RapidOCR, Docling, Pydantic.

### Environment Variables (`.env.example`)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `PINECONE_API_KEY` | No | Pinecone key (leave empty for local ChromaDB) |
| `TAVILY_API_KEY` | No | Tavily web search API key |
| `LANGSMITH_API_KEY` | No | LangSmith observability |
| `LANGSMITH_TRACING` | No | Enable/disable LangSmith tracing |
| `GEMINI_API_KEY` | No | Google Gemini (secondary LLM) |
| `PUSHOVER_*` | No | Push notification service |

### Custom Exceptions (`src/core/exceptions.py`)

```
AIProjectError (base)
├── RetrievalError
│   ├── VectorStoreConnectionError
│   └── DocumentProcessingError
└── AgentError
    ├── LLMResponseError
    └── RelevanceAuditError
```

---

## 9. Evaluation Pipeline

### Framework

**File:** `evaluation/eval_pipeline.py`
**Framework:** RAGAS (Retrieval-Augmented Generation Assessment)
**LLM for evaluation:** `gpt-4o-mini` via OpenAI AsyncOpenAI client

### Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Is the retrieved context relevant to the question? |
| **Context Recall** | Does the retrieval cover all necessary information? |

### Configuration

- Default dataset: `evaluation/datasets/golden_set_v2.json`
- Max samples: 10 (configurable via `--max-samples`)
- Timeout: 300 seconds per evaluation
- Max retries: 10
- Output: `evaluation/results/report_YYYYMMDD_HHMM.csv`

### Golden Datasets

| File | Q&A Pairs | Topics |
|------|-----------|--------|
| `golden_set_v1.json` | ~25 | DeepSeek, Google, NVIDIA |
| `golden_set_v2.json` | ~32 | + Cloudflare, infrastructure metrics, environmental data |

Each entry has `question` and `ground_truth` fields.

### Latest Results

| Metric | Score |
|--------|-------|
| Faithfulness | 90.75% |
| Answer Relevancy | 95.91% |
| Context Precision | 67.21% |
| Context Recall | 90.00% |

---

## 10. Deployment

### Option 1: Local Development

```bash
# Terminal 1 — API backend
make api
# or: uvicorn src.api.app:app --reload --port 8001

# Terminal 2 — Streamlit UI
make ui
# or: streamlit run ui/streamlit_frontend.py --server.port 8501

# Terminal 3 (optional) — CLI
make chat
# or: uv run python -m src.main
```

### Option 2: Docker

```bash
# Build and start all services
docker compose up --build

# Services:
#   API      → http://localhost:8001
#   UI       → http://localhost:8501
#   API docs → http://localhost:8001/docs
```

**Docker Compose services:**

| Service | Command | Port | Notes |
|---------|---------|------|-------|
| `api` | `uvicorn src.api.app:app --host 0.0.0.0 --port 8001` | 8001 | Healthcheck on `/api/health` |
| `ui` | `streamlit run ui/streamlit_frontend.py --server.port 8501` | 8501 | Depends on API (healthy) |

Both services load environment from `.env` and mount `./data` as a volume.

### Option 3: UV (direct)

```bash
uv sync
uv run python -m src.main
```

---

## 11. Development Tools

### Makefile Targets

| Command | Purpose |
|---------|---------|
| `make install` | Install dependencies |
| `make api` | Run FastAPI backend (port 8001) |
| `make ui` | Run Streamlit UI (port 8501) |
| `make chat` | Run interactive CLI |
| `make ingest` | Trigger document ingestion via API |
| `make eval` | Run RAGAS evaluation pipeline |
| `make lint` | Run Ruff linter |
| `make format` | Format code with Ruff |
| `make typecheck` | MyPy type checking |
| `make test` | Run pytest |
| `make docker-up` | Start Docker containers |
| `make docker-down` | Stop Docker containers |
| `make clean` | Remove __pycache__, logs, etc. |

### Code Quality Tools

- **Ruff** — linting and formatting.
- **MyPy** — static type checking.
- **Pytest** — test runner (no tests written yet).

### Utility Scripts (`misc/`)

- `generate_workflow_diagram.py` — generates PNG Mermaid diagram of the LangGraph workflow.
- `env_setup.txt` — Windows UV/Python environment setup guide.
- `git.txt` — Git workflow and commit convention reference.
- `terminal.txt` — Quick command reference for running services.
- `results_eval.txt` — Evaluation results log.

---

## 12. Dependencies

### Core Dependencies

| Category | Packages |
|----------|----------|
| **LLM** | `langchain`, `langchain-openai`, `langchain-core`, `langchain-community` |
| **Orchestration** | `langgraph` (workflow), `langsmith` (tracing) |
| **Vector Stores** | `chromadb`, `pinecone`, `langchain-pinecone` |
| **Retrieval** | `rank-bm25`, `langchain-text-splitters` |
| **Documents** | `docling`, `pymupdf`, `langchain-docling` |
| **API** | `fastapi`, `uvicorn`, `httpx` |
| **UI** | `streamlit` |
| **Validation** | `pydantic`, `pydantic-settings` |
| **Data** | `pandas` |
| **Evaluation** | `ragas` |
| **Logging** | `python-json-logger` |

### Dev Dependencies

| Package | Purpose |
|---------|---------|
| `pytest` | Testing |
| `ruff` | Linting & formatting |
| `mypy` | Type checking |
| `ipykernel` | Jupyter notebook support |

### Requirements

- **Python:** >= 3.12
- **Package Manager:** `uv` (recommended) or `pip`
- **Lock File:** `uv.lock` for reproducible installs

---

## 13. Data

### Source Documents (`data/raw/`)

7 PDF documents (~111 MB total):

| Document | Size | Topic |
|----------|------|-------|
| DeepSeek Technical Report.pdf | 1.3 MB | DeepSeek-R1 RL training methodology |
| google-2024-environmental-report.pdf | 15 MB | Google environmental/sustainability metrics |
| Impact-Report-2025-Final.pdf | 13 MB | Cloudflare impact report |
| NASDAQ_NVDA_2024.pdf | 34 MB | NVIDIA financial report (10-K) |
| NVIDIA-2025-Annual-Report.pdf | 48 MB | NVIDIA annual report |
| the-openai-nonprofit-commission-report.pdf | 1.1 MB | OpenAI nonprofit commission |

### Cached Data (`data/cache/`)

Processed chunks stored as JSON (~10.6 MB total):
- One JSON file per source PDF (contains chunked `page_content` + `metadata`).
- `bm25_documents.json` (6.3 MB) — BM25 keyword search index.

### Chunk Metadata Format

```json
{
  "page_content": "...",
  "metadata": {
    "source": "DeepSeek Technical Report.pdf",
    "file_path": "data/raw/DeepSeek Technical Report.pdf",
    "total_pages": 22,
    "page": 0,
    "parser": "pymupdf",
    "chunk_size": 935,
    "chunk_index": 0,
    "chunk_hash": "sha256..."
  }
}
```

---

## 14. Honest Project Audit

> This section is a fair, evidence-based critique of the project as it stands today. It is intentionally not a "yes-man" review — its purpose is to make the next iteration of the project better, not to make the current one feel good. Findings are grounded in direct code reads (file:line references where applicable).

### 14.1 What's Done Well

| Area | Why it's good | Evidence |
|------|---------------|----------|
| **Clean module boundaries** | `api/`, `core/`, `engines/`, `retrieval/`, `workflow/` each have one responsibility. Easy to navigate, easy to swap implementations. | Folder layout under [src/](src/) |
| **LangGraph state machine is explicit** | Conditional routing (`decide_after_relevance`, `decide_after_verification`) is declarative and inspectable; the self-correction loop is bounded by `max_iterations`. | [src/workflow/orchestrator.py](src/workflow/orchestrator.py) |
| **Structured outputs everywhere** | Every agent uses `.with_structured_output(PydanticModel)`. No fragile regex parsing of LLM text. | [src/schemas/agent_schemas.py](src/schemas/agent_schemas.py), [src/workflow/agents/](src/workflow/agents/) |
| **Dual-parser PDF strategy** | Docling primary with a 300s timeout and 80-page guard, PyMuPDF fallback if Docling produces fewer than 5 chunks. Robust against bad PDFs. | [src/retrieval/document_processor.py](src/retrieval/document_processor.py) |
| **Content-addressed chunk cache** | SHA256 of `filename + content` deduplicates chunks across re-ingests and acts as the vector store ID. Idempotent ingestion. | [src/retrieval/document_processor.py](src/retrieval/document_processor.py), [src/retrieval/vector_store.py](src/retrieval/vector_store.py) |
| **Hybrid retrieval done right** | BM25 + dense vectors via `EnsembleRetriever`, weights externalised to YAML, BM25 index persisted to disk. | [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py) |
| **Configuration discipline** | Pydantic `BaseSettings` merges YAML + `.env` with validation. No magic strings in business code. | [src/core/config_loader.py](src/core/config_loader.py), [config/settings.yaml](config/settings.yaml) |
| **Custom exception hierarchy** | `RetrievalError`, `AgentError`, etc. — callers can catch specific failure modes instead of bare `Exception`. | [src/core/exceptions.py](src/core/exceptions.py) |
| **Docker compose with healthchecks** | API container has a real `/api/health` healthcheck and the UI `depends_on` it as `service_healthy`. Cold-start ordering is correct. | [docker-compose.yml](docker-compose.yml) |
| **Honest evaluation pipeline** | RAGAS with four metrics on a versioned golden dataset. The reported scores (Faithfulness 90.75%, Context Precision 67.21%) are not cherry-picked — Context Precision is the lowest, and that's reported anyway. | [evaluation/eval_pipeline.py](evaluation/eval_pipeline.py) |
| **Secrets handling is correct (despite first impressions)** | `.env` is properly listed in `.gitignore` and is **not** in the repo. Only `.env.example` is committed. | [.gitignore](.gitignore), `git ls-files` |

### 14.2 What's Not So Good

| Issue | Severity | Where | Why it matters |
|-------|----------|-------|----------------|
| **Zero tests** | High | `tests/` does not exist; `pytest` is in `dependencies` but unused | The whole self-correction loop, the dual-parser fallback, the relevance routing — none of it is regression-protected. Any refactor is a coin flip. |
| **No CI/CD** | High | No `.github/workflows/`, no `azure-pipelines.yml` | Lint/type/test never run automatically. A broken `main` is one push away. |
| **Dockerfile runs as root** | Medium | [Dockerfile](Dockerfile) (no `USER` directive) | Container escape -> host root. Trivial fix: add a non-root user. |
| **Single-stage Docker build** | Medium | [Dockerfile](Dockerfile) | Image ships `build-essential`, `pip` cache, dev deps, source tree, and notebooks. Multi-stage would slim it dramatically. |
| **CORS wide open + no auth + no rate-limit** | High | [src/api/app.py](src/api/app.py), [src/api/routes.py](src/api/routes.py), `cors_origins: ["*"]` in [config/settings.yaml](config/settings.yaml) | Anyone on the network can hit `/api/chat` and burn OpenAI credits. Anyone can `/api/ingest`. This is fine for `localhost`, unsafe anywhere else. |
| **LangSmith tracing only wired into the CLI** | Medium | Configured in `.env` but only consumed by [src/main.py](src/main.py), not by the FastAPI app | The thing you actually want to trace (production requests) is invisible. |
| **Singleton without a lock** | Medium | `_system` global at the bottom of [src/api/dependencies.py](src/api/dependencies.py) | Two concurrent startup requests can race-initialise the pipeline twice. Unlikely under uvicorn `--workers 1`, real under multi-worker. |
| **Sync file I/O inside async paths** | Low–Medium | BM25 cache load/save in [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py); blocking `input()` in [src/main.py](src/main.py) | Won't show up at low load, but will pin the event loop under concurrency. Use `aiofiles` or run in a thread. |
| **Generic `except Exception:` swallows context** | Low–Medium | [src/retrieval/document_processor.py](src/retrieval/document_processor.py), [src/api/routes.py](src/api/routes.py) | The original traceback is buried in logs; clients see opaque 500s. Re-raise with `from e`. |
| **Magic numbers scattered through agents** | Low | `_DOCLING_TIMEOUT_SECONDS = 300`, `_MAX_DOCLING_PAGES = 80`, `_MAX_CONTEXT_CHARS = 12_000` (duplicated in two agents), `_MIN_ANSWER_LENGTH = 5` | Tuning requires editing source. These should live in `settings.yaml`. |
| **`pyproject.toml` declares tools but never configures them** | Low | [pyproject.toml](pyproject.toml) has no `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` | `make lint` runs Ruff with default rules; `make typecheck` runs MyPy in its loosest mode. The tools exist on paper. |
| **Dev deps mixed with runtime deps** | Low | `mypy`, `ruff`, `pytest`, `ipykernel` all in main `[dependencies]` | The production image installs Jupyter and a type-checker. Move them to `[project.optional-dependencies].dev`. |
| **`uv.lock` is gitignored** | Low | [.gitignore](.gitignore) line 72 | Defeats the point of a lock file. `uv.lock` should be committed for reproducible builds; only `.uv/` cache should be ignored. |
| **README is thin** | Low | [README.md](README.md) | Quick-start works, but no architecture, no troubleshooting, no failure modes, no production guidance. New contributors will lean entirely on this `PROJECT_SUMMARY.md`. |
| **No request correlation IDs** | Low | [src/api/routes.py](src/api/routes.py) | When a chat request fans out across 4 LangGraph nodes and 2 retries, you cannot grep one transaction out of the JSON log. |
| **No prompt-injection guardrails** | Low | [src/workflow/agents/researcher.py](src/workflow/agents/researcher.py), [src/workflow/agents/verifier.py](src/workflow/agents/verifier.py) | User questions and document content are concatenated into prompts without sanitisation. Low-risk for an internal tool, real risk if exposed. |

### 14.3 What's Missing Outright

These aren't "weaknesses" — they simply don't exist yet.

1. **A test suite.** Even three smoke tests (one per agent) would catch the majority of refactor regressions. Start with a `test_orchestrator.py` that asserts a `NO_MATCH` short-circuits at `check_relevance`.
2. **Authentication.** API key header, OAuth2 bearer, or Microsoft Entra ID — pick one. `chat` and `ingest` should require it; `health` should not.
3. **Rate limiting.** `slowapi` (Starlette/FastAPI middleware) is the smallest possible win — per-IP and per-key limits in 30 lines.
4. **Error tracking.** Sentry or Azure Application Insights. Right now, exceptions land in a JSON log file nobody is watching.
5. **Metrics and tracing.** OpenTelemetry -> OTLP -> Application Insights / Grafana. At minimum: request latency, token usage per request, retrieval hit count, iteration count distribution, vector store latency.
6. **An async ingestion path.** `/api/ingest` is synchronous and long-running — a 100MB PDF blocks the request thread for minutes. Push it onto a queue (Service Bus, Azure Queue Storage, or Celery+Redis) and return a job ID.
7. **A response and embedding cache.** Redis-keyed by `(question_hash, vector_store_revision)` for full responses; by `chunk_hash` for embeddings. Embeddings are deterministic — recomputing them on every ingest is pure waste.
8. **A circuit breaker around upstream APIs.** OpenAI/Pinecone outages currently surface as raw 500s. `pybreaker` or `tenacity` with exponential backoff, plus a "degraded mode" that returns cached answers.
9. **CI/CD pipeline.** Lint -> typecheck -> test -> build -> scan -> deploy. Even if only the first three exist on day one, that's enough.
10. **Pre-commit hooks.** Ruff format, secrets scan (`detect-secrets`), trailing-whitespace. Zero-cost quality floor.
11. **A real backup story for the vector store.** Pinecone has snapshots; ChromaDB lives in a directory that nobody is backing up.
12. **A troubleshooting runbook.** "Pinecone returns 503 — what now?" "Ingestion failed mid-document — what's the recovery?" "Verification keeps looping — how do I debug it?"
13. **Cost guardrails.** A per-day OpenAI spend cap, alerts on token-usage anomalies, model-fallback enforcement.

### 14.4 Verdict

The project is a **well-architected proof of concept**. The agentic loop is the right idea, executed with good engineering taste — explicit state, structured outputs, hybrid retrieval, dual-parser ingestion, evaluation pipeline. For a learning / demo / internal tool, it's above average.

It is **not production-ready**. The two highest-leverage changes, in order, are:
1. **Write tests** (you cannot safely change anything without them).
2. **Add auth + rate-limiting + observability** (you cannot safely *deploy* anything without them).

Everything else in §14.2 and §14.3 is downstream of those two.

---

## 15. Scaling & Production Deployment on Microsoft Azure

> This section describes a target architecture for taking the project from "runs on my laptop" to "runs in production for real users on Azure". The strategy is **cloud-native, managed-first** — every component is replaced with a managed Azure equivalent so the team owns business logic, not infrastructure.

### 15.1 Target Architecture

```
                                   ┌────────────────────────┐
                                   │   Azure Front Door     │
                                   │  (global LB + WAF)     │
                                   └───────────┬────────────┘
                                               │
                                   ┌───────────▼────────────┐
                                   │   API Management       │
                                   │  (auth, rate-limit,    │
                                   │   quotas, OpenAPI)     │
                                   └───────────┬────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
            ┌──────────▼──────────┐                       ┌────────────▼─────────┐
            │  Container Apps     │                       │  Container Apps      │
            │  • API (FastAPI)    │◄──── internal ───────►│  • UI (Streamlit)    │
            │  • KEDA autoscale   │                       │  • Same env / image  │
            └──────────┬──────────┘                       └──────────────────────┘
                       │
       ┌───────────────┼─────────────────┬──────────────────┬─────────────────┐
       │               │                 │                  │                 │
┌──────▼──────┐ ┌──────▼──────┐  ┌───────▼────────┐ ┌───────▼───────┐ ┌───────▼────────┐
│ Azure       │ │ Azure AI    │  │ Azure Blob     │ │ Azure Cosmos  │ │ Azure Service  │
│ OpenAI      │ │ Search      │  │ Storage        │ │ DB (NoSQL)    │ │ Bus            │
│ • gpt-4o-   │ │ • hybrid    │  │ • raw PDFs     │ │ • chat        │ │ • async        │
│   mini      │ │   (BM25 +   │  │ • parsed       │ │   sessions    │ │   ingestion    │
│ • text-embed│ │   vector)   │  │   chunks       │ │ • audit log   │ │   queue        │
│   -3-small  │ │ • semantic  │  │ • eval results │ │               │ │                │
└─────────────┘ │   ranker    │  └────────────────┘ └───────────────┘ └────────────────┘
                └─────────────┘

Cross-cutting:
┌────────────────┐ ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Azure Key      │ │ Application     │ │ Azure Container  │ │ Microsoft Entra  │
│ Vault          │ │ Insights        │ │ Registry (ACR)   │ │ ID (Azure AD)    │
│ • all secrets  │ │ • traces        │ │ • signed images  │ │ • workload IDs   │
│ • CSI driver   │ │ • metrics       │ │ • Trivy scans    │ │ • user auth      │
└────────────────┘ │ • LLM token use │ └──────────────────┘ └──────────────────┘
                   └─────────────────┘
```

### 15.2 Component-by-Component Mapping

| Local component | Azure replacement | Why |
|-----------------|-------------------|-----|
| OpenAI direct API | **Azure OpenAI Service** | Same models (`gpt-4o-mini`, `text-embedding-3-small`), but with Azure-native auth (managed identity), regional data residency, private endpoints, cost-centre tagging, and PTU options for predictable latency. |
| Pinecone / ChromaDB | **Azure AI Search** | Native hybrid search (BM25 + vector + semantic ranker) means `HybridSearcher` collapses into one `search` call. Removes BM25 cache file entirely. Integrated vector indexing, integrated security. |
| `data/raw/` PDFs | **Azure Blob Storage** (hot tier) | Durable, versioned, lifecycle-managed (cool/archive after N days). The `document_processor` reads from a blob URL instead of a local path. Triggers Azure Functions / Service Bus on upload. |
| `data/cache/` chunks | **Azure Blob Storage** (cool tier) or skip entirely (Azure AI Search becomes the cache) | Cheap, durable, versioned. Or remove the chunk cache layer entirely once AI Search owns chunk storage. |
| `_system` singleton in-process | **Azure Container Apps revision** | Container Apps handles scale-to-zero, KEDA-based autoscaling on HTTP request count, blue/green via revisions. The "singleton" lives per-replica, which is fine because the vector store is external. |
| Streamlit UI | **Azure Container Apps** (separate app) | Same image, different ingress. Or migrate to Azure Static Web Apps if/when the UI moves to React/Next. |
| `.env` file | **Azure Key Vault** + **Container Apps secrets** | Secrets injected as env vars at runtime via the Key Vault CSI driver or native Container Apps secret references. Rotation without redeploy. |
| JSON file logging | **Azure Application Insights** + **Log Analytics** | Structured logs, distributed traces (with OpenTelemetry), Kusto queries, alerts. Replace `python-json-logger` with `opentelemetry-sdk` + `azure-monitor-opentelemetry`. |
| LangSmith (optional) | **Keep LangSmith** *or* migrate to App Insights | LangSmith is still the best LLM-specific trace UI; keep it side-by-side with App Insights. Wire it into the FastAPI app, not just the CLI. |
| Synchronous `/api/ingest` | **Azure Service Bus** + **Azure Container Apps Job** | Upload PDF -> Blob -> Event Grid -> Service Bus -> Container Apps Job runs `document_processor` -> upserts into AI Search. The HTTP endpoint returns a job ID immediately. |
| Chat history (none) | **Azure Cosmos DB for NoSQL** | Per-session message log, partition key = `session_id`. TTL-based retention for GDPR. |
| No CI/CD | **GitHub Actions** *or* **Azure DevOps Pipelines** | Build -> test -> Trivy scan -> push to ACR -> deploy revision to Container Apps. OIDC federation, no long-lived secrets. |
| No auth | **Microsoft Entra ID** + **API Management** | API Management validates JWTs at the edge; FastAPI trusts the validated identity in headers. End users sign in via Entra ID; service-to-service uses managed identities. |
| No rate-limiting | **API Management policies** | Per-subscription quotas and per-IP throttling defined declaratively in APIM policy XML. No code change needed. |
| No backup | **Blob soft-delete + AI Search index snapshots** | Built-in. |

### 15.3 Why Azure AI Search Is the Single Biggest Win

The current `HybridSearcher` is ~80 lines of code that loads BM25 from a JSON file, builds a `BM25Retriever`, builds a `VectorStoreRetriever`, wraps them in `EnsembleRetriever`, and worries about cache invalidation. **Azure AI Search does all of this server-side**, plus a learned semantic ranker on top, plus filters, plus security trimming. The migration looks like:

```python
# Before: src/retrieval/hybrid_search.py (~80 lines)
self.retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=self.weights,
)

# After (sketch):
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

results = search_client.search(
    search_text=question,                     # BM25
    vector_queries=[VectorizedQuery(          # dense vector
        vector=embed(question),
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )],
    query_type="semantic",                    # learned re-ranker
    semantic_configuration_name="default",
    top=top_k,
)
```

This deletes `hybrid_search.py`, the BM25 JSON cache, and the in-process index rebuild logic.

### 15.4 Recommended Phased Rollout

**Phase 1 — Lift & shift**
- Build a multi-stage Dockerfile, run as non-root, push to **Azure Container Registry**.
- Deploy API + UI as two **Azure Container Apps** (single revision, single replica).
- Move secrets to **Azure Key Vault**, reference from Container Apps secret store.
- Wire **Application Insights** via `azure-monitor-opentelemetry` — instrument FastAPI automatically.
- Set up **GitHub Actions** with OIDC federation: lint -> test (once tests exist) -> build -> scan -> deploy.
- Front the API with **API Management**, enforce a subscription key on `/api/chat` and `/api/ingest`. `/api/health` stays anonymous.

*Outcome:* same architecture, running on managed Azure services, with secrets, observability, and a deployment pipeline.

**Phase 2 — Replace the data plane**
- Provision an **Azure OpenAI** resource in the same region; swap the OpenAI client to use `AzureChatOpenAI` and `AzureOpenAIEmbeddings`. Authenticate with managed identity, not keys.
- Provision **Azure AI Search** (Standard tier minimum for semantic ranker). Re-index existing chunks. Replace `HybridSearcher` with a thin `AzureAISearchRetriever`. Delete the BM25 JSON cache.
- Move raw PDFs into **Azure Blob Storage** (`raw-pdfs` container). `DocumentProcessor` reads from a blob URL.

*Outcome:* zero local state. The compute is fully stateless and horizontally scalable.

**Phase 3 — Productionise the request path**
- Add **Azure Cosmos DB** for chat sessions (partition key `session_id`, TTL on the container).
- Add **Microsoft Entra ID** authentication via APIM (JWT validation policy). End users get a real login.
- Add **Azure Cache for Redis** for response caching keyed by question hash + index version. TTL 1h. Cuts OpenAI cost on repeat queries.
- Enable **KEDA** scaling on the API Container App: scale on HTTP concurrency (target 10 in-flight requests per replica), min 1 / max 10 replicas.

*Outcome:* the system survives a launch.

**Phase 4 — Async ingestion + resilience**
- Move `/api/ingest` to a fire-and-forget pattern: HTTP POST -> enqueue Service Bus message -> return `202 Accepted` with a job ID.
- A **Container Apps Job** consumes the queue, runs the existing `DocumentProcessor` end-to-end, writes results to AI Search, and updates job status in Cosmos DB.
- Add circuit breakers (`pybreaker`) around Azure OpenAI and AI Search. On open circuit, return cached responses with a "degraded mode" header.
- Configure **Azure Front Door** with WAF in front of APIM for global presence, DDoS protection, and TLS termination.

*Outcome:* ingestion no longer blocks the API; upstream failures degrade gracefully.

**Phase 5 — Hardening (ongoing)**
- Cost dashboards in Azure Cost Management; budgets and anomaly alerts on Azure OpenAI spend.
- Private endpoints on every PaaS resource; no public network access.
- Workload identities everywhere, zero secrets in environment variables.
- Disaster recovery: paired-region replication for Blob, geo-redundant Cosmos DB, periodic AI Search index snapshots into Blob.

### 15.5 Estimated Monthly Cost (Order of Magnitude, Low Traffic)

| Service | Tier | Approx. cost (USD/mo) |
|---------|------|-----------------------|
| Azure Container Apps (2 apps × 1 replica × 0.5 vCPU × 1 GiB) | Consumption | ~$30 |
| Azure OpenAI (`gpt-4o-mini` + embeddings, ~1M tokens/mo) | PAYG | ~$15–$30 |
| Azure AI Search | Basic | ~$75 |
| Azure Blob Storage (10 GB hot) | LRS | ~$1 |
| Azure Cosmos DB (serverless) | Serverless | ~$5 |
| Application Insights (5 GB ingest) | Pay-as-you-go | ~$15 |
| API Management | Consumption | ~$3 (per million calls) |
| Key Vault | Standard | ~$1 |
| Container Registry | Basic | ~$5 |
| **Total** | | **~$150–$200/mo** |

> AI Search Basic is the floor cost. A serious launch should plan for AI Search Standard (~$250/mo) and Azure OpenAI PTU if predictable latency matters.

### 15.6 What NOT to Do

- **Do not deploy this to a single Azure VM with `docker compose`.** That throws away every benefit of the cloud and recreates every operational problem you had on the laptop.
- **Do not keep ChromaDB in production on a mounted Azure File share.** It works for one replica and breaks the moment you scale out. Use AI Search.
- **Do not put OpenAI/Pinecone keys in App Service application settings.** Use Key Vault references or, better, managed identity to Azure OpenAI.
- **Do not skip API Management because "FastAPI can do auth too".** APIM gives you rate-limiting, quotas, IP filtering, JWT validation, request transformation, and a developer portal — none of which are worth re-implementing in middleware.
- **Do not migrate all five phases at once.** Phase 1 alone (lift & shift to Container Apps + Key Vault + App Insights + CI/CD) delivers 80% of the operational value at 20% of the effort.

---

## Audit Methodology

Sections 14 and 15 of this document were generated by reading the source code directly — `src/`, `config/`, `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, `.gitignore`, and `README.md` — rather than relying on any other documentation. Specific findings reference `file:line` where the underlying behaviour lives. The Azure architecture in §15 is opinionated; substitute equivalents (AKS instead of Container Apps, Cognitive Search at a different tier, etc.) as the team's preferences and budget dictate.

---

## Key Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Singleton** | `SystemManager` | One-time pipeline initialization per app lifecycle |
| **LangGraph State Machine** | `RAGOrchestrator` | Type-safe, declarative workflow with conditional routing |
| **Structured Output** | All agents | Pydantic models via `.with_structured_output()` for reliable parsing |
| **Hybrid Retrieval** | `HybridSearcher` | BM25 + Vector ensemble for keyword and semantic coverage |
| **Self-Correcting Loop** | Orchestrator verify node | Verification failure triggers re-research (up to 3 iterations) |
| **Dual-Parser Fallback** | `DocumentProcessor` | Docling primary, PyMuPDF fallback for robustness |
| **Content-Addressed Caching** | Document chunks | SHA256 hash deduplication + JSON cache to avoid reprocessing |
| **Dependency Injection** | FastAPI `Depends()` | Clean component wiring for routes |
| **Async/Await** | All I/O operations | Non-blocking LLM calls and retrieval |
| **Abstract Base Classes** | `BaseLLM`, `BaseEmbeddingModel` | Swappable engine implementations |
