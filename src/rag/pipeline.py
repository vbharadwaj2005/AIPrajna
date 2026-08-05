from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint

from src.config import (
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    _MAX_INPUT_LENGTH,
    logger,
)

_RAG_PROMPT = """You are a compliance analyst assistant. Answer the question based strictly on the provided context.

Rules:
- If the context lacks sufficient information, say "I cannot answer this based on the provided documents."
- Always cite the source document name for each fact you reference.
- Be concise and precise.

Context:
{context}

Question: {question}

Answer:"""


class LlmService:
    _instance: HuggingFaceEndpoint | None = None

    def _get_llm(self) -> HuggingFaceEndpoint:
        if LlmService._instance is None:
            if not HUGGINGFACE_API_KEY:
                raise ValueError(
                    "HUGGINGFACE_API_KEY is not set. "
                    "Get a free token at https://huggingface.co/settings/tokens"
                )
            LlmService._instance = HuggingFaceEndpoint(
                repo_id=HUGGINGFACE_MODEL,
                huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                temperature=0.1,
                max_new_tokens=512,
            )
            logger.info("LLM ready — model: %s", HUGGINGFACE_MODEL)
        return LlmService._instance

    def generate(self, prompt: str) -> str:
        llm = self._get_llm()
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)


llm_service = LlmService()


def _format_context(documents: List[Tuple[Document, float]]) -> str:
    parts: List[str] = []
    for idx, (doc, score) in enumerate(documents, start=1):
        source = doc.metadata.get("source_file", "unknown")
        parts.append(f"[Source {idx}: {source} (relevance: {score:.3f})]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_query(question: str, documents: List[Tuple[Document, float]]) -> str:
    context = _format_context(documents)

    if len(context) > _MAX_INPUT_LENGTH:
        logger.warning("Context truncated from %d to %d chars", len(context), _MAX_INPUT_LENGTH)
        context = context[:_MAX_INPUT_LENGTH]

    prompt = _RAG_PROMPT.format(context=context, question=question)
    return llm_service.generate(prompt)


def extract_sources(documents: List[Tuple[Document, float]]) -> List[dict]:
    seen: set[str] = set()
    sources: List[dict] = []

    for doc, score in documents:
        source_file = doc.metadata.get("source_file", "unknown")
        if source_file in seen:
            continue
        seen.add(source_file)

        content = doc.page_content
        excerpt = content[:300]
        if len(content) > 300:
            excerpt += "..."

        sources.append({
            "file": source_file,
            "path": doc.metadata.get("source_path", ""),
            "relevance": round(score, 3),
            "excerpt": excerpt,
        })

    return sources
