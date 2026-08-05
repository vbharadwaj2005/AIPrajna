from __future__ import annotations

import streamlit as st

from src.config import logger
from src.embedding.embedder import embedding_service
from src.rag.pipeline import answer_query, extract_sources
from src.retrieval.reranker import reranker


def render_sources(sources: list[dict]) -> None:
    with st.expander("View sources", expanded=False):
        for source in sources:
            label = (
                f"**{source['file']}**  "
                f"<span class='source-badge'>relevance: {source['relevance']}</span>"
            )
            st.markdown(label, unsafe_allow_html=True)
            st.text(source["excerpt"])
            st.divider()


def render_chat_interface() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

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
                render_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
