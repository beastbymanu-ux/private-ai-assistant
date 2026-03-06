FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY config.py .
COPY ingest.py .
COPY rag_engine.py .
COPY app.py .
COPY setup.sh .

# Create directories
RUN mkdir -p /app/documents /app/chroma_db

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entry point: ingest documents then start UI
CMD ["sh", "-c", "python ingest.py --docs-dir /app/documents && streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"]
