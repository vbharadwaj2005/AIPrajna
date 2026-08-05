"""
DocInsight — Vector Embedding Service.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL, logger


class EmbeddingService:
    """Singleton service for generating dense text vector embeddings."""

    _instance: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if EmbeddingService._instance is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            EmbeddingService._instance = SentenceTransformer(EMBEDDING_MODEL)
        return EmbeddingService._instance

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        model = self._get_model()
        logger.info("Embedding %d text(s)...", len(texts))
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        return model.encode([query], normalize_embeddings=True)[0]


embedding_service = EmbeddingService()
