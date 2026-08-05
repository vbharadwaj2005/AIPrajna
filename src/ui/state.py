from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    defaults: dict[str, object] = {
        "retriever": None,
        "vectorstore": None,
        "documents": [],
        "dense_index": None,
        "messages": [],
        "index_built": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
