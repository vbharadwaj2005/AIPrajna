"""
DocInsight — Semantic Document Chunking & Text Splitting.
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

from core.config import CHUNK_OVERLAP, CHUNK_SIZE, SEMANTIC_THRESHOLD, logger
from core.embedder import embedding_service


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _merge_related_chunks(chunks: list[Document]) -> list[Document]:
    logger.info("Merging semantically similar chunks (threshold=%.2f)...", SEMANTIC_THRESHOLD)
    texts = [c.page_content for c in chunks]
    embeddings: np.ndarray = embedding_service.embed_texts(texts)

    merged: list[Document] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        current_emb = embeddings[idx]
        next_idx = idx + 1

        while next_idx < len(chunks):
            similarity = cosine_similarity([current_emb], [embeddings[next_idx]])[0][0]
            if similarity > SEMANTIC_THRESHOLD:
                current.page_content += "\n\n" + chunks[next_idx].page_content
                for key, value in chunks[next_idx].metadata.items():
                    current.metadata.setdefault(key, value)
                next_idx += 1
            else:
                break

        current.metadata["chunk_id"] = len(merged)
        merged.append(current)
        idx = next_idx

    logger.info("Semantic merging: %d → %d chunks", len(chunks), len(merged))
    return merged


def semantic_chunk(documents: list[Document]) -> list[Document]:
    """Execute recursive character splitting followed by semantic embedding merging."""
    if not documents:
        return []

    splitter = _build_splitter()
    chunks = splitter.split_documents(documents)
    logger.info("Recursive split: %d → %d chunks", len(documents), len(chunks))
    return _merge_related_chunks(chunks)
