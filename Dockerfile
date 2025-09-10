# -------- Base image (slim) --------
FROM python:3.11-slim AS base

# Ensure clean Python output and no .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: curl for healthcheck, gosu to drop privileges at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gosu ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# -------- Dependency layer (leverages Docker cache) --------
# If you use Poetry/uv instead, swap this block accordingly.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# -------- App layer --------
COPY . /app

# Create an unprivileged user
RUN adduser --disabled-password --gecos "" appuser

# Create local persistent store for vector/metadata DBs and ensure ownership
RUN mkdir -p /app/.memdb && chown -R appuser:appuser /app

# Defaults (you can override via env/compose)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WEB_CONCURRENCY=2 \
    LOG_LEVEL=INFO \
    MEMDB_DIR=/app/.memdb

# Expose API port
EXPOSE 8000


# Healthcheck pings the docs route
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/docs" >/dev/null || exit 1

# Run Uvicorn pointing at your app module: api/memory_api.py -> app
CMD ["sh", "-c", "uvicorn api.memory_api:app --host ${HOST} --port ${PORT} --workers ${WEB_CONCURRENCY} --log-level ${LOG_LEVEL} --proxy-headers --timeout-keep-alive 75"]

