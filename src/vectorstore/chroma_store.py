from __future__ import annotations

from typing import List

import chromadb
import numpy as np
from langchain_core.documents import Document

from src.config import CHROMA_DIR, logger


class ChromaStore:
    def __init__(self, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready at %s", collection_name, CHROMA_DIR)

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        ids: List[str] = []
        metadatas: List[dict] = []
        texts: List[str] = []

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

    def similarity_search(self, query_embedding: np.ndarray, k: int = 10) -> List[Document]:
        results = self._collection.query(
            query_embeddings=query_embedding.reshape(1, -1).tolist(),
            n_results=k,
        )

        return [
            Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection.name)
            logger.info("Deleted collection '%s'", self._collection.name)
        except Exception as exc:
            logger.warning("Failed to delete collection: %s", exc)
