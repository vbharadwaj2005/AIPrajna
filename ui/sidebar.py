"""
DocInsight — Sidebar Document Management & Index Control.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from core.chroma_store import ChromaStore
from core.chunker import semantic_chunk
from core.config import COLLECTION_NAME, DATA_DIR, HUGGINGFACE_MODEL
from core.embedder import embedding_service
from core.hybrid_search import HybridRetriever
from core.loader import load_document, load_documents_from_dir


def render_sidebar() -> None:
    """Render the document management sidebar."""
    with st.sidebar:
        st.header("Documents")
        upload_tab, folder_tab, status_tab = st.tabs(["Upload", "From Folder", "Status"])

        with upload_tab:
            _render_upload_tab()

        with folder_tab:
            _render_folder_tab()

        with status_tab:
            _render_status_tab()

        st.divider()
        st.caption(f"Model: {HUGGINGFACE_MODEL.split('/')[-1]}")

        if st.session_state.messages:
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


def _render_upload_tab() -> None:
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            all_docs: list[Document] = []
            for uploaded in uploaded_files:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    docs = load_document(tmp_path)
                    for doc in docs:
                        doc.metadata["source_file"] = uploaded.name
                    all_docs.extend(docs)
                except Exception as exc:
                    st.error(f"Failed to load {uploaded.name}: {exc}")
                finally:
                    os.unlink(tmp_path)

            st.session_state.documents = semantic_chunk(all_docs)
            st.session_state.index_built = False
            st.success(f"Loaded {len(st.session_state.documents)} chunk(s)")


def _render_folder_tab() -> None:
    st.markdown("Folder: `data/documents/` ")
    if st.button("Load from Folder", use_container_width=True):
        with st.spinner("Loading documents..."):
            raw_docs = load_documents_from_dir(DATA_DIR)
            if not raw_docs:
                st.warning("No supported documents found in data/documents/")
            else:
                st.session_state.documents = semantic_chunk(raw_docs)
                st.session_state.index_built = False
                st.success(f"Loaded {len(st.session_state.documents)} chunk(s)")


def _render_status_tab() -> None:
    st.metric("Chunks in memory", len(st.session_state.documents))
    db_status = "Active" if st.session_state.vectorstore else "Not built"
    st.metric("Vector store", db_status)
    if st.session_state.vectorstore:
        st.metric("Indexed vectors", st.session_state.vectorstore.count())

    if not st.session_state.index_built and st.session_state.documents:
        if st.button("Build Vector Index", use_container_width=True, type="primary"):
            with st.spinner("Embedding and indexing..."):
                texts = [d.page_content for d in st.session_state.documents]
                embeddings = embedding_service.embed_texts(texts)

                store = ChromaStore(COLLECTION_NAME)
                store.add_documents(st.session_state.documents, embeddings)

                st.session_state.vectorstore = store
                st.session_state.dense_index = embeddings
                st.session_state.retriever = HybridRetriever(
                    st.session_state.documents
                )
                st.session_state.index_built = True
                st.success("Index built!")
                st.rerun()
    elif st.session_state.index_built:
        st.info("Index ready for queries")
        if st.button("Rebuild Index", use_container_width=True):
            st.session_state.index_built = False
            st.rerun()
