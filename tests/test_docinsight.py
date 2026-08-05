from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# Define Document class first
class Document:
    def __init__(self, page_content: str, metadata: dict | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}

# Mock third-party dependencies if not installed
for module_name in [
    "numpy",
    "sentence_transformers",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_core",
    "langchain_core.documents",
    "langchain_huggingface",
    "langchain_text_splitters",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "chromadb",
    "rank_bm25",
    "streamlit",
]:
    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except ImportError:
            mod = MagicMock()
            if module_name == "langchain_core.documents":
                mod.Document = Document
            sys.modules[module_name] = mod

# Override Document in sys.modules if it's a MagicMock
if hasattr(sys.modules.get("langchain_core.documents"), "Document"):
    if isinstance(getattr(sys.modules["langchain_core.documents"], "Document"), MagicMock):
        sys.modules["langchain_core.documents"].Document = Document

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    HYBRID_WEIGHT_DENSE,
    MAX_INPUT_LENGTH,
    SEMANTIC_THRESHOLD,
    TOP_K_INITIAL,
    TOP_K_RERANKED,
)
from src.ingestion.loader import SUPPORTED_EXTENSIONS, _validate_path
from src.rag.pipeline import _format_context, extract_sources
from src.ui.state import init_session_state


class TestConfig(unittest.TestCase):
    def test_config_defaults(self) -> None:
        self.assertEqual(CHUNK_SIZE, 512)
        self.assertEqual(CHUNK_OVERLAP, 64)
        self.assertEqual(SEMANTIC_THRESHOLD, 0.75)
        self.assertEqual(TOP_K_INITIAL, 20)
        self.assertEqual(TOP_K_RERANKED, 5)
        self.assertEqual(COLLECTION_NAME, "docinsight")
        self.assertEqual(MAX_INPUT_LENGTH, 4096)
        self.assertTrue(0 <= HYBRID_WEIGHT_DENSE <= 1)


class TestIngestion(unittest.TestCase):
    def test_supported_extensions(self) -> None:
        self.assertIn(".pdf", SUPPORTED_EXTENSIONS)
        self.assertIn(".txt", SUPPORTED_EXTENSIONS)
        self.assertIn(".docx", SUPPORTED_EXTENSIONS)

    def test_validate_path_nonexistent(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _validate_path("non_existent_file_xyz.txt")


class TestRagPipeline(unittest.TestCase):
    def test_format_context(self) -> None:
        docs = [
            (Document(page_content="Policy overview text.", metadata={"source_file": "policy.pdf"}), 0.95),
            (Document(page_content="Compliance rules.", metadata={"source_file": "rules.docx"}), 0.82),
        ]
        context = _format_context(docs)
        self.assertIn("[Source 1: policy.pdf (relevance: 0.950)]", context)
        self.assertIn("Policy overview text.", context)
        self.assertIn("[Source 2: rules.docx (relevance: 0.820)]", context)
        self.assertIn("Compliance rules.", context)

    def test_extract_sources(self) -> None:
        docs = [
            (Document(page_content="First chunk content", metadata={"source_file": "doc1.txt", "source_path": "/path/1"}), 0.9),
            (Document(page_content="Second chunk content from same doc", metadata={"source_file": "doc1.txt", "source_path": "/path/1"}), 0.85),
            (Document(page_content="Third chunk content", metadata={"source_file": "doc2.txt", "source_path": "/path/2"}), 0.7),
        ]
        sources = extract_sources(docs)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["file"], "doc1.txt")
        self.assertEqual(sources[0]["relevance"], 0.9)
        self.assertEqual(sources[1]["file"], "doc2.txt")
        self.assertEqual(sources[1]["relevance"], 0.7)


class TestUiState(unittest.TestCase):
    @patch("src.ui.state.st")
    def test_init_session_state(self, mock_st) -> None:
        mock_session_state = {}
        mock_st.session_state = mock_session_state
        init_session_state()
        self.assertIn("retriever", mock_session_state)
        self.assertIn("vectorstore", mock_session_state)
        self.assertIn("documents", mock_session_state)
        self.assertIn("dense_index", mock_session_state)
        self.assertIn("messages", mock_session_state)
        self.assertIn("index_built", mock_session_state)
        self.assertFalse(mock_session_state["index_built"])


if __name__ == "__main__":
    unittest.main()
