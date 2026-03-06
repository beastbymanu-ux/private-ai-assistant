"""
RAG Engine — Retrieval-Augmented Generation.

Connects ChromaDB (local vector store) with Ollama (local LLM)
to answer questions based on ingested documents.

The entire pipeline runs locally:
- Embeddings: sentence-transformers (local)
- Vector search: ChromaDB (local)
- LLM inference: Ollama (local)
- Result: Zero data leaves the machine.
"""
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)


# System prompt for the AI assistant
ASSISTANT_PROMPT = PromptTemplate(
    template="""You are a precise AI assistant answering questions based on internal documents.

Use ONLY the context provided below to answer. If the context does not contain
enough information to answer the question, say so explicitly.

Do not make up information. Do not use external knowledge.
Cite the source document when possible.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    All components run locally:
    - HuggingFace embeddings (sentence-transformers)
    - ChromaDB vector store
    - Ollama LLM (Llama 3 / Mistral)
    """

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize all RAG components."""
        try:
            logger.info("Initializing RAG engine (all local)...")

            # Local embeddings
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
            )

            # Local vector store
            logger.info(f"Loading vector store from: {CHROMA_PERSIST_DIR}")
            self.vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
            )

            # Verify vector store has data
            collection_count = self.vector_store._collection.count()
            if collection_count == 0:
                logger.warning("Vector store is empty. Run ingest.py first.")
                return False
            logger.info(f"Vector store loaded: {collection_count} chunks indexed")

            # Local LLM via Ollama
            logger.info(f"Connecting to Ollama: {OLLAMA_BASE_URL} (model: {OLLAMA_MODEL})")
            self.llm = OllamaLLM(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.1,
            )

            # Build QA chain
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": TOP_K_RESULTS},
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": ASSISTANT_PROMPT},
            )

            self._initialized = True
            logger.info("RAG engine initialized successfully. All components local.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            return False

    def query(self, question: str) -> dict:
        """
        Query the RAG pipeline.

        Args:
            question: The user's question

        Returns:
            dict with:
                - answer: str
                - sources: list of source documents
                - chunks_used: int
        """
        if not self._initialized:
            return {
                "answer": "RAG engine not initialized. Run initialize() first.",
                "sources": [],
                "chunks_used": 0,
            }

        try:
            result = self.qa_chain.invoke({"query": question})

            # Extract source information
            sources = []
            seen_sources = set()
            for doc in result.get("source_documents", []):
                source_name = doc.metadata.get("source", "Unknown")
                if source_name not in seen_sources:
                    seen_sources.add(source_name)
                    sources.append({
                        "document": source_name,
                        "content_preview": doc.page_content[:200] + "...",
                    })

            return {
                "answer": result["result"],
                "sources": sources,
                "chunks_used": len(result.get("source_documents", [])),
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "chunks_used": 0,
            }

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        if not self._initialized:
            return {"status": "not initialized"}

        count = self.vector_store._collection.count()
        return {
            "status": "active",
            "chunks_indexed": count,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": OLLAMA_MODEL,
            "ollama_url": OLLAMA_BASE_URL,
            "top_k": TOP_K_RESULTS,
            "data_locality": "100% local — zero external API calls",
        }
