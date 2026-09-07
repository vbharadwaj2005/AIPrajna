"""
AIPrajna — Compliance Multi-Document RAG Platform (Streamlit)
Main entry point for running the Streamlit application.
"""

from __future__ import annotations

import streamlit as st

from ui import init_session_state, render_chat_interface, render_sidebar

st.set_page_config(
    page_title="AIPrajna — Compliance RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_STYLES = """
<style>
    .stApp { background-color: #f8f9fa; }
    .source-badge {
        background-color: #e8eaf6;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.75rem;
    }
    footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)

with st.container():
    st.title("AIPrajna")
    st.caption("Compliance RAG — Ask questions across your documents with source-grounded answers")

init_session_state()
render_sidebar()
render_chat_interface()
