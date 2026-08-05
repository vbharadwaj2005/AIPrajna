from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import streamlit as st
from langchain_core.documents import Document

from src.config import (
    COLLECTION_NAME,
    DATA_DIR,
    HUGGINGFACE_MODEL,
    logger,
)
from src.embedding.embedder import embedding_service
from src.ingestion.chunker import semantic_chunk
from src.ingestion.loader import load_document, load_documents_from_dir
from src.rag.pipeline import answer_query, extract_sources, llm_service
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import reranker
from src.vectorstore.chroma_store import ChromaStore

st.set_page_config(
    page_title="DocInsight",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
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
st.markdown(_CSS, unsafe_allow_html=True)

with st.container():
    st.title("DocInsight")
    st.caption(
        "Compliance RAG \u2014 Ask questions across your documents "
        "with source-grounded answers"
    )

# ── Session initialisation ─────────────────────────────────────────
_DEFAULT_SESSION: Dict[str, object] = {
    "retriever": None,
    "vectorstore": None,
    "documents": [],
    "dense_index": None,
    "messages": [],
    "index_built": False,
}
for key, default in _DEFAULT_SESSION.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar: document management ───────────────────────────────────
with st.sidebar:
    st.header("Documents")

    upload_tab, folder_tab, status_tab = st.tabs(["Upload", "From Folder", "Status"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Choose PDF, TXT, or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                all_docs: List[Document] = []
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

    with folder_tab:
        st.markdown(f"Folder: `data/documents/`")
        if st.button("Load from Folder", use_container_width=True):
            with st.spinner("Loading documents..."):
                raw_docs = load_documents_from_dir(DATA_DIR)
                if not raw_docs:
                    st.warning("No supported documents found in data/documents/")
                else:
                    st.session_state.documents = semantic_chunk(raw_docs)
                    st.session_state.index_built = False
                    st.success(f"Loaded {len(st.session_state.documents)} chunk(s)")

    with status_tab:
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

    st.divider()
    st.caption(f"Model: {HUGGINGFACE_MODEL.split('/')[-1]}")

    if st.session_state.messages:
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Chat display ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources", expanded=False):
                for source in msg["sources"]:
                    label = (
                        f"**{source['file']}**  "
                        f"<span class='source-badge'>relevance: {source['relevance']}</span>"
                    )
                    st.markdown(label, unsafe_allow_html=True)
                    st.text(source["excerpt"])
                    st.divider()

# ── Chat input / query flow ────────────────────────────────────────
if not st.session_state.index_built:
    st.info(
        "Upload documents via the sidebar and build the vector index "
        "to start asking questions."
    )
elif prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                query_embedding = embedding_service.embed_query(prompt)
                initial_results = st.session_state.retriever.retrieve(
                    prompt, query_embedding, st.session_state.dense_index
                )
                reranked = reranker.rerank(prompt, initial_results)
                answer = answer_query(prompt, reranked)
                sources = extract_sources(reranked)
            except Exception as exc:
                logger.exception("Query failed")
                answer = f"An error occurred: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("View sources", expanded=False):
                for source in sources:
                    label = (
                        f"**{source['file']}**  "
                        f"<span class='source-badge'>relevance: {source['relevance']}</span>"
                    )
                    st.markdown(label, unsafe_allow_html=True)
                    st.text(source["excerpt"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
