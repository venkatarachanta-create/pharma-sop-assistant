import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents))

import streamlit as st
from src.rag.pipeline import run_pharma_rag
from src.config.settings import settings
from src.ingestion.loader import load_and_chunk_pdfs
from src.rag.vectorstore import build_vectorstore

st.set_page_config(page_title="Pharma SOP Compliance Assistant", page_icon="💊", layout="wide")
st.title("💊 Pharma SOP & Regulatory Compliance Assistant")
st.caption("Grounded Regulatory RAG • Input/Output Guardrails • Compliance Audit Trail")


def _ensure_index():
    index_file = settings.chroma_persist_dir / "index.faiss"
    if not index_file.exists():
        with st.spinner("Indexing CDSCO regulatory PDFs into FAISS (one-time setup)..."):
            chunks = load_and_chunk_pdfs()
            build_vectorstore(chunks)


_ensure_index()

with st.sidebar:
    st.subheader("System Status")
    st.write(f"**Provider:** `{settings.llm_provider.upper()}`")
    st.write(f"**Model:** `{settings.llm_model}`")
    st.write(f"**Top-K Chunks:** `{settings.top_k_results}`")
    indexed_pdfs = list(settings.raw_pdf_dir.glob("*.pdf"))
    st.write(f"**Loaded PDFs:** `{len(indexed_pdfs)}`")
    for p in indexed_pdfs:
        st.caption(f"• {p.name}")

    st.divider()
    if st.button("🔄 Re-Index Documents"):
        if settings.chroma_persist_dir.exists():
            for item in settings.chroma_persist_dir.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()
        with st.spinner("Re-indexing PDFs..."):
            chunks = load_and_chunk_pdfs()
            build_vectorstore(chunks)
        st.success(f"Indexed {len(chunks)} chunks!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

if user_input := st.chat_input("Ask a regulatory or SOP compliance question..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Verifying regulatory documents..."):
            result = run_pharma_rag(user_input)
            if result["blocked"]:
                st.warning(result["answer"])
            else:
                st.markdown(result["answer"])
                with st.expander("🔍 View Retrieved Regulatory Evidence"):
                    for src, ctx in zip(result["sources"], result["contexts"]):
                        st.markdown(f"**{src}**")
                        st.text(ctx)

    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
