"""
Document Ingestion Pipeline.

Loads documents from the /documents directory, splits them into chunks,
generates embeddings locally, and stores them in ChromaDB.

No data leaves the machine. All processing is local.

Usage:
    python ingest.py
    python ingest.py --docs-dir /path/to/documents
"""
import os
import sys
import argparse
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DOCS_DIR,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Supported file types and their loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
}


def discover_documents(docs_dir: str) -> list:
    """Find all supported documents in the directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)

    files = []
    for ext in LOADER_MAP:
        found = list(docs_path.rglob(f"*{ext}"))
        files.extend(found)
        if found:
            logger.info(f"Found {len(found)} {ext} files")

    if not files:
        logger.warning(f"No supported documents found in {docs_dir}")
        logger.info(f"Supported formats: {', '.join(LOADER_MAP.keys())}")
        sys.exit(0)

    return files


def load_documents(files: list) -> list:
    """Load all documents using appropriate loaders."""
    all_docs = []

    for file_path in files:
        ext = file_path.suffix.lower()
        loader_class = LOADER_MAP.get(ext)

        if not loader_class:
            logger.warning(f"Skipping unsupported file: {file_path}")
            continue

        try:
            loader = loader_class(str(file_path))
            docs = loader.load()

            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["full_path"] = str(file_path)

            all_docs.extend(docs)
            logger.info(f"Loaded: {file_path.name} ({len(docs)} pages/sections)")

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    return all_docs


def split_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """Create ChromaDB vector store with local embeddings."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (local)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    logger.info(f"Creating vector store at: {CHROMA_PERSIST_DIR}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    logger.info(f"Vector store created with {len(chunks)} embeddings")
    return vector_store


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("--docs-dir", default=DOCS_DIR, help="Path to documents directory")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRIVATE AI ASSISTANT — Document Ingestion")
    logger.info("All processing runs locally. No data leaves this machine.")
    logger.info("=" * 60)

    # Step 1: Discover documents
    files = discover_documents(args.docs_dir)
    logger.info(f"Found {len(files)} documents to process")

    # Step 2: Load documents
    documents = load_documents(files)
    logger.info(f"Loaded {len(documents)} document sections")

    # Step 3: Split into chunks
    chunks = split_documents(documents)

    # Step 4: Create vector store with local embeddings
    vector_store = create_vector_store(chunks)

    logger.info("=" * 60)
    logger.info("Ingestion complete.")
    logger.info(f"Documents: {len(files)}")
    logger.info(f"Chunks indexed: {len(chunks)}")
    logger.info(f"Vector store: {CHROMA_PERSIST_DIR}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL} (local)")
    logger.info("Ready for queries.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
