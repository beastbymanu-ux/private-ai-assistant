"""
Configuration for Private AI Assistant.
All settings loaded from environment variables for security.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Paths
DOCS_DIR = os.getenv("DOCS_DIR", "./documents")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Embedding model (runs locally via sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
