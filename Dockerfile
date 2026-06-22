# ── Stage: Runtime ────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps for sentence-transformers + torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside container
WORKDIR /app

# Copy requirements first — Docker layer cache skips pip install
# if requirements.txt hasn't changed
COPY requirements.txt .

# Install dependencies directly into container (no venv needed)
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker marks container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the gateway
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
