from __future__ import annotations

import streamlit as st

from src.ui.chat import render_chat_interface
from src.ui.sidebar import render_sidebar
from src.ui.state import init_session_state

st.set_page_config(
    page_title="DocInsight",
    page_icon="\U0001f50d",
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
    st.title("DocInsight")
    st.caption("Compliance RAG — Ask questions across your documents with source-grounded answers")

init_session_state()
render_sidebar()
render_chat_interface()
