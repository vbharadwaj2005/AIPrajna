"""
DocInsight — ChromaDB Persistent Vector Store Engine.
"""

from __future__ import annotations

import chromadb
import numpy as np
from langchain_core.documents import Document

from core.config import CHROMA_DIR, logger


class ChromaStore:
    """Interface wrapper around ChromaDB persistent vector collection."""

    def __init__(self, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready at %s", collection_name, CHROMA_DIR)

    def add_documents(self, documents: list[Document], embeddings: np.ndarray) -> None:
        ids: list[str] = []
        metadatas: list[dict] = []
        texts: list[str] = []

        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id", idx)
            source = doc.metadata.get("source_file", "unknown")
            ids.append(f"chunk_{chunk_id}_{source}")
            metadatas.append({
                "source_file": source,
                "source_path": doc.metadata.get("source_path", ""),
                "chunk_id": str(chunk_id),
            })
            texts.append(doc.page_content)

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )
        logger.info("Indexed %d vectors in ChromaDB", len(documents))

    def count(self) -> int:
        return self._collection.count()
