from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL, TOP_K_RERANKED, logger


class Reranker:
    _instance: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if Reranker._instance is None:
            logger.info("Loading cross-encoder: %s", CROSS_ENCODER_MODEL)
            Reranker._instance = CrossEncoder(CROSS_ENCODER_MODEL)
        return Reranker._instance

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        scored = [
            (doc, float(score))
            for (doc, _), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        return scored[:TOP_K_RERANKED]


reranker = Reranker()
