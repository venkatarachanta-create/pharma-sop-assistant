# Use official lightweight Python 3.10 image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system build dependencies needed by ChromaDB / ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (caches layer for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and sample PDFs
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/raw_pdfs/ ./data/raw_pdfs/
COPY data/eval_datasets/ ./data/eval_datasets/

# Create directories for persistent vector DB and audit logs
RUN mkdir -p data/chroma_db data/logs

# Expose Streamlit port
EXPOSE 8501

# Healthcheck so Docker knows when Streamlit is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run ingestion if chroma_db is empty, then start Streamlit
CMD ["sh", "-c", "python scripts/ingest_docs.py && streamlit run src/ui/app.py --server.port=8501 --server.address=0.0.0.0"]
