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
              Vector Store           LLM (OpenAI)         Web Search
           (Pinecone / Chroma)        GPT-4o-mini          (Tavily)
```

**Agents:** Relevance Checker → Researcher → Verifier (orchestrated via LangGraph)

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker & Docker Compose (for containerised run)
- API keys — copy `.env.example` to `.env` and fill in your keys

## Quick Start — Local

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/agentic-chatbot.git
cd agentic-chatbot

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
# Edit .env and fill in your API keys

# 5. Run the API backend
uvicorn src.api.app:app --reload --port 8001

# 6. Run the Streamlit UI (in a separate terminal)
streamlit run ui/streamlit_frontend.py

# 7. (Optional) Run the CLI
uv run python -m src.main
```

## Quick Start — Docker

```bash
# Copy and fill in secrets
copy .env.example .env   # then edit .env

# Build and start all services
docker compose up --build

# Services:
#   API      → http://localhost:8001
#   UI       → http://localhost:8501
#   API docs → http://localhost:8001/docs
```

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
│   ├── retrieval/   # Document processor, hybrid search, vector store
│   ├── schemas/     # Pydantic schemas
│   └── workflow/    # LangGraph orchestrator, agents, tools, memory
├── ui/              # Streamlit frontend
├── .env.example     # Template for required environment variables
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Environment Variables

See [.env.example](.env.example) for all required variables.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `PINECONE_API_KEY` | Pinecone key — leave empty to use local ChromaDB |
| `TAVILY_API_KEY` | Tavily web search API key |
| `LANGSMITH_API_KEY` | LangSmith observability (optional) |
| `GEMINI_API_KEY` | Google Gemini (optional, for secondary LLM) |
