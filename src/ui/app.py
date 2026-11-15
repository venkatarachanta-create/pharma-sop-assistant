import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents))

import streamlit as st
from src.rag.pipeline import run_pharma_rag
from src.config.settings import settings
from src.ingestion.loader import load_and_chunk_pdfs
from src.rag.vectorstore import build_vectorstore, load_vectorstore

st.set_page_config(page_title="Pharma SOP Compliance Assistant", page_icon="💊", layout="wide")
st.title("💊 Pharma SOP & Regulatory Compliance Assistant")
st.caption("Grounded Regulatory RAG • Input/Output Guardrails • Compliance Audit Trail")


def _ensure_index():
    vs = load_vectorstore()
    if vs._collection.count() == 0:
        with st.spinner("Indexing CDSCO regulatory PDFs into ChromaDB (one-time setup)..."):
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

    if st.button("🔄 Re-Index Documents"):
        if settings.chroma_persist_dir.exists():
            for item in settings.chroma_persist_dir.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()
        with st.spinner("Re-indexing PDFs..."):
            chunks = load_and_chunk_pdfs()
            build_vectorstore(chunks)
        st.success(f"Indexed {len(chunks)} chunks!")
