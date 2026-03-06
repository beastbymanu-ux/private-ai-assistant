#!/bin/bash
# ============================================================
# Private AI Assistant — Quick Setup
#
# This script:
# 1. Starts Ollama + Assistant containers
# 2. Pulls the Llama 3 model
# 3. Ingests sample documents
# 4. Opens the web UI
#
# Prerequisites: Docker, Docker Compose
# ============================================================

set -e

echo "============================================"
echo "  Private AI Assistant — Setup"
echo "  All data stays on this machine."
echo "============================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    echo "Install it from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Create documents directory if it doesn't exist
if [ ! -d "./documents" ]; then
    mkdir -p ./documents
    echo "Created ./documents directory."
    echo "Add your PDF, TXT, MD, or CSV files to this directory."
fi

# Check if documents exist
DOC_COUNT=$(find ./documents -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" -o -name "*.csv" \) 2>/dev/null | wc -l)
if [ "$DOC_COUNT" -eq 0 ]; then
    echo ""
    echo "WARNING: No documents found in ./documents/"
    echo "Add your files there before running this script."
    echo "Supported formats: PDF, TXT, MD, CSV"
    echo ""
    echo "Example: cp ~/my-docs/*.pdf ./documents/"
    echo ""
    read -p "Continue anyway with empty knowledge base? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo ""
echo "[1/4] Starting Docker containers..."
docker compose up -d

echo ""
echo "[2/4] Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
    echo "  Waiting for Ollama..."
done
echo "  Ollama is ready."

echo ""
echo "[3/4] Pulling Llama 3 model (this may take a few minutes on first run)..."
docker exec private-ai-ollama ollama pull llama3

echo ""
echo "[4/4] System is ready."
echo ""
echo "============================================"
echo "  Access the AI Assistant:"
echo "  http://localhost:8501"
echo ""
echo "  Architecture:"
echo "  - LLM: Llama 3 via Ollama (local)"
echo "  - Embeddings: sentence-transformers (local)"
echo "  - Vector DB: ChromaDB (local disk)"
echo "  - External API calls: ZERO"
echo "============================================"

# Open browser (macOS)
if command -v open &> /dev/null; then
    open http://localhost:8501
fi
