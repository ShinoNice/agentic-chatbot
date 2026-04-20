.PHONY: install api ui chat ingest eval lint format typecheck test cov docker-up docker-down clean

# ── Environment ───────────────────────────────────────────────────────

install:
	uv sync --frozen

# ── Run Services ──────────────────────────────────────────────────────

api:
	uvicorn src.api.app:app --reload --port 8001

ui:
	streamlit run ui/streamlit_frontend.py --server.port 8501

chat:
	python -m src.main

# ── Data Pipeline ─────────────────────────────────────────────────────

ingest:
	python -c "import asyncio; from src.api.dependencies import SystemManager; s=SystemManager(); print(asyncio.run(s.ingest()))"

eval:
	python evaluation/eval_pipeline.py

# ── Code Quality ──────────────────────────────────────────────────────

lint:
	ruff check src/ ui/ evaluation/

format:
	ruff format src/ ui/ evaluation/

typecheck:
	mypy src/

test:
	uv run pytest -q --no-header

cov:
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=60

# ── Docker ────────────────────────────────────────────────────────────

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

# ── Cleanup ───────────────────────────────────────────────────────────

clean:
	python -c "import shutil,pathlib;[shutil.rmtree(p) for d in ['__pycache__','.mypy_cache','.pytest_cache','.ruff_cache'] for p in pathlib.Path('.').rglob(d) if p.is_dir()]"
