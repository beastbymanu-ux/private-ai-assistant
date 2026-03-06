"""
Private AI Assistant — Web Interface.

Streamlit-based chat UI for querying internal documents
using a fully local RAG pipeline.

Architecture:
    User → Streamlit UI → RAG Engine → ChromaDB + Ollama → Response
    (All local. Zero external API calls.)
"""
import streamlit as st
import time
import logging
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Private AI Assistant",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .privacy-badge {
        background: linear-gradient(135deg, #0f9b0f 0%, #0a7d0a 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .stats-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .source-card {
        background: #f0f4f8;
        border-left: 3px solid #4a90d9;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZE RAG ENGINE (cached)
# ============================================================
@st.cache_resource
def get_rag_engine():
    """Initialize RAG engine once and cache it."""
    engine = RAGEngine()
    success = engine.initialize()
    return engine, success


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="privacy-badge">🔒 100% LOCAL</div>', unsafe_allow_html=True)
    st.markdown("### System Status")

    engine, initialized = get_rag_engine()

    if initialized:
        stats = engine.get_stats()
        st.success("System Active")

        st.markdown(f"""
        <div class="stats-card">
            <strong>LLM:</strong> {stats['llm_model']}<br>
            <strong>Embeddings:</strong> {stats['embedding_model']}<br>
            <strong>Chunks indexed:</strong> {stats['chunks_indexed']:,}<br>
            <strong>Top-K results:</strong> {stats['top_k']}<br>
            <strong>Data locality:</strong> {stats['data_locality']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("System Not Ready")
        st.warning("Run `python ingest.py` to index documents first.")

    st.markdown("---")
    st.markdown("### Architecture")
    st.code(
        "User Query\n"
        "  → Embedding (local)\n"
        "  → ChromaDB search (local)\n"
        "  → Ollama / Llama 3 (local)\n"
        "  → Response\n"
        "\n"
        "External API calls: 0\n"
        "Data leaving server: None",
        language=None,
    )

    st.markdown("---")
    st.markdown("### Privacy Guarantees")
    st.markdown("""
    - ✅ All inference runs on local hardware
    - ✅ Embeddings generated locally
    - ✅ Vector store on local disk
    - ✅ No telemetry or tracking
    - ✅ GDPR compliant by architecture
    - ✅ Works fully air-gapped
    """)


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.markdown('<p class="main-header">🔒 Private AI Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Ask questions about your internal documents. '
    'All processing runs locally — zero data leaves this server.'
    '</p>',
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📎 Sources ({len(message['sources'])} documents)"):
                for source in message["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{source["document"]}</strong><br>'
                        f'{source["content_preview"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    if not initialized:
        response_text = (
            "The system is not initialized. "
            "Please run `python ingest.py` to index your documents first."
        )
        sources = []
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                start_time = time.time()
                result = engine.query(prompt)
                elapsed = time.time() - start_time

                response_text = result["answer"]
                sources = result["sources"]

                st.markdown(response_text)

                # Show performance metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"⏱️ {elapsed:.1f}s | 📄 {result['chunks_used']} chunks used")
                with col2:
                    st.caption("🔒 Processed 100% locally")

                # Show sources
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)} documents)"):
                        for source in sources:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{source["document"]}</strong><br>'
                                f'{source["content_preview"]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    })
