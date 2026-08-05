"""
DocInsight — Hybrid Dense + BM25 Sparse Retrieval Engine.
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.config import HYBRID_WEIGHT_DENSE, TOP_K_INITIAL, logger


class HybridRetriever:
    """Weighted fusion of dense vector cosine similarity and BM25 keyword scores."""

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("At least one document is required to build the retriever")

        self._documents = documents
        self._texts = [doc.page_content for doc in documents]
        self._bm25 = self._build_bm25_index()
        logger.info("HybridRetriever ready — %d documents indexed", len(documents))

    def _build_bm25_index(self) -> BM25Okapi:
        tokenized_corpus = [text.lower().split() for text in self._texts]
        return BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        dense_index: np.ndarray,
    ) -> list[tuple[Document, float]]:
        dense_scores = np.dot(dense_index, query_embedding)
        d_min, d_max = dense_scores.min(), dense_scores.max()
        if d_max - d_min > 1e-8:
            dense_scores = (dense_scores - d_min) / (d_max - d_min)

        bm25_scores = np.array(self._bm25.get_scores(query.lower().split()))
        b_min, b_max = bm25_scores.min(), bm25_scores.max()
        if b_max - b_min > 1e-8:
            bm25_scores = (bm25_scores - b_min) / (b_max - b_min)

        combined = (
            HYBRID_WEIGHT_DENSE * dense_scores
            + (1.0 - HYBRID_WEIGHT_DENSE) * bm25_scores
        )
        top_indices = np.argsort(combined)[::-1][:TOP_K_INITIAL]

        return [(self._documents[idx], float(combined[idx])) for idx in top_indices]
