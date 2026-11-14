import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from src.rag.pipeline import run_pharma_rag
from src.config.settings import settings

st.set_page_config(page_title="Pharma SOP Compliance Assistant", page_icon="💊", layout="wide")
st.title("💊 Pharma SOP & Regulatory Compliance Assistant")
st.caption("Grounded Regulatory RAG • Input/Output Guardrails • Compliance Audit Trail")

with st.sidebar:
    st.subheader("System Status")
    st.write(f"**Provider:** `{settings.llm_provider.upper()}`")
    st.write(f"**Model:** `{settings.llm_model}`")
    st.write(f"**Top-K Chunks:** `{settings.top_k_results}`")
    indexed_pdfs = list(settings.raw_pdf_dir.glob("*.pdf"))
    st.write(f"**Loaded PDFs:** `{len(indexed_pdfs)}`")
    for p in indexed_pdfs:
        st.caption(f"• {p.name}")

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
