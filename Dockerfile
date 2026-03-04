# ─── Base Image ───────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ─── System Dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python Dependencies ─────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── Copy Project Source ──────────────────────────────────────────────────────
COPY . .

# ─── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ─── Default Port Exposure ───────────────────────────────────────────────────
# Actual port binding is controlled by docker-compose per service
EXPOSE 8001
EXPOSE 8501
