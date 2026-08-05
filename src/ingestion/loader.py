from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from src.config import logger

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx"}


def _validate_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{resolved.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return resolved


def _pick_loader(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".txt":
        return TextLoader(str(path), encoding="utf-8")
    return Docx2txtLoader(str(path))


def load_document(file_path: str | Path) -> list[Document]:
    path = _validate_path(file_path)
    loader = _pick_loader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source_file", path.name)
        doc.metadata.setdefault("source_path", str(path))
    logger.info("Loaded %s — %d page(s)", path.name, len(docs))
    return docs


def load_documents_from_dir(directory: str | Path) -> list[Document]:
    resolved = Path(directory).resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")

    all_docs: list[Document] = []
    for ext in SUPPORTED_EXTENSIONS:
        for fpath in sorted(resolved.glob(f"*{ext}")):
            try:
                all_docs.extend(load_document(fpath))
            except Exception as exc:
                logger.warning("Skipping %s: %s", fpath.name, exc)
    return all_docs
